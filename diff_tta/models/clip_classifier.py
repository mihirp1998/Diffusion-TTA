import clip
import torch
import torch.nn as nn
import tqdm
print("Torch version:", torch.__version__)


class ClipClassifier(nn.Module):
    def __init__(self, classes, class_arch):
        super().__init__()

        imagenet_templates = [
            'a photo of a {}.',
        ]
        if class_arch == "clipr50":
            model, _ = clip.load("RN50",jit=False)
        elif class_arch == "clipr101":
            model, _ = clip.load("RN101",jit=False)
        elif class_arch == "clipb32":
            model, _ = clip.load("ViT-B/32",jit=False)
        elif class_arch == "clipb16":
            model, _ = clip.load("ViT-B/16",jit=False)
        elif class_arch == "clipl14":
            model, _ = clip.load("ViT-L/14",jit=False)


        self.final_fc = nn.Linear(768,1000,bias=False)
        with torch.no_grad():
            zeroshot_weights = zeroshot_classifier(classes, imagenet_templates, model)
        self.final_fc.weight.data = zeroshot_weights.T
        self.model = model

    def forward(self, images):
        image_features = self.model.encode_image(images)
        logits = 100. * self.final_fc(image_features)
        return logits

def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm.tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights



def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


if __name__ == "__main__":
    from dataset.dataset_class_label import ImageNetDataset

    _, preprocess = clip.load("ViT-L/14",jit=False)

    dataset = ImageNetDataset(classification_transform=preprocess,
                              diffusion_transform=preprocess)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=2)
    clip_classifier = ClipClassifier(dataset.class_names)

    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for i, batch in enumerate(iter(dataloader)):
            images = batch['image_disc'].cuda()
            target = batch['class_idx'].cuda()

            logits = clip_classifier(images)

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)
            if i ==5:
                break

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100

    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-5 accuracy: {top5:.2f}")
