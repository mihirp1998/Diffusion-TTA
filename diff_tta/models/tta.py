import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import diff_tta.utils as utils


class TTABase_Class(nn.Module):
    """Define a general interface for TTA, which is applicable for discrete
    sampling and gradient-descent based TTA.

    Args:
        unet_model: A nn.Module indicates the U-Net in the diffusion model
        class_model: A nn.Module indicates the classifier
        scheduler: A nn.Module indicates the scheduler for the diffusion model
        diffusion: A nn.Module indicates the diffusion model
        class_text_embeddings: A tensor of shape [num_classes, embedding_dim]
        classes: A tensor of shape [num_selected_classes]. For example,
                 ImageNet-R/A only uses 200 classes out of 1000 classes
        config: A config object
    """
    def __init__(self, unet_model, class_model, scheduler, diffusion,
                 class_text_embeddings, classes, config):
        super().__init__()
        self.unet_model = unet_model
        self.class_model = class_model
        self.scheduler = scheduler
        self.class_text_embeddings = class_text_embeddings
        self.diffusion_model = diffusion
        self.config = config

        self.classes = classes

        self.before_tta_acc = []
        self.after_tta_acc = []

        if config.model.use_dit:
            self.class_text_embeddings = (
                self.unet_model.y_embedder.embedding_table.weight
            )
        
        if self.classes is not None:
            self.class_text_embeddings = nn.Parameter(
                self.class_text_embeddings[self.classes]
            )

    def evaluate(self, batch, before_tta=False, after_tta=False):
        """Implement this function in subclasses.
        """
        raise NotImplementedError

    def diffusion_timesteps(self):
        return self.diffusion_model.num_timesteps

    def _unet_pred_noise(self, x_start, t, noise, context):
        """A helper function to predict noise using the U-Net

        Args:
            x_start: A tensor of shape [1, C, H, W]
            t: A tensor of shape [num_timesteps]
            noise: A tensor of shape [num_timesteps, C, H, W]
            context: A tensor of shape [1, C] or [1, num_tokens, C]

        Returns:
            pred_noise: A tensor of shape [num_timesteps, C, H, W]
        """
        device = t.device
        nt = t.shape[0]
        if self.config.model.use_dit:
            x_start = x_start.expand(nt, -1, -1, -1)
            noised_latent = self.diffusion_model.q_sample(
                x_start=x_start, t=t, noise=noise
            )

            model_output = self.unet_model(
                noised_latent, t, y=None, context=context.expand(nt, -1)
            )

            C = noised_latent.shape[1]
            pred_noise, _ = torch.split(model_output, C, dim=1)
        else:
            alphas_cumprod = self.scheduler.alphas_cumprod.to(device)

            noised_latent = (
                x_start * (alphas_cumprod[t]**0.5).view(-1, 1, 1, 1).to(device)
                + noise * ((1 - alphas_cumprod[t])**0.5).view(-1, 1, 1, 1).to(device)
            )

            pred_noise = self.unet_model(
                noised_latent,
                t,
                encoder_hidden_states=context.expand(nt, -1, -1)
            ).sample

        return pred_noise

    def classify(self, image):
        """A helper function to outputs classification results

        Args:
            image: A tensor of shape [1, 3, H, W]

        Returns:
            logits: A tensor of shape [batch_size, num_classes]. If `self.classes`
                    is not None, `logits` will be a tensor of shape
                    [len(self.classes)]
            topk_idx: A tensor of shape [batch_size, K]. If `self.adapt_topk` is
                      -1, set K to 5. This index correlates to the ordering in
                      `outputs`.
            max_class_idx: A tensor of shape [batch_sizes, 1]. This index
                           correlates to the ordering in the number of classes
                           of the pre-trained classifier.
            topk_class_idx: A tensor of shape [batch_size, K]. If
                            `self.adpt_topk` is -1, set K to 5. This index
                            correlates to the ordering in the number of classes
                            of the pre-trained classifier.
        """
        # Classify with the classifier
        logits = self.class_model(image)

        # Remove unused classes (in ImageNet)
        if self.classes is not None:
            logits = logits[:, self.classes]

        # Pick top-K classes
        probs = logits.softmax(-1)
        max_idx = probs.argmax(-1)

        K = probs.shape[-1] if self.config.tta.adapt_topk == -1 else self.config.tta.adapt_topk
        topk_idx = probs.argsort(descending=True)[:, :K]

        if self.classes is not None:
            classes = torch.tensor(self.classes).to(logits.device)
            max_class_idx = classes[max_idx.flatten()].view(max_idx.shape)
            topk_class_idx = classes[topk_idx.flatten()].view(topk_idx.shape)
        else:
            max_class_idx, topk_class_idx = max_idx, topk_idx

        return logits, topk_idx, max_class_idx, topk_class_idx


    def forward(self, image, x_start=None, t=None, noise=None, pred_top_idx=None):
        """Perform classification or compute diffusion loss.

        Args:
            image: A tensor of shape [1, 3, H, W]
            x_start: A tensor of shape [1, C, latent_H, latent_W]
            t: A tensor of shape [num_timesteps]
            noise: A tensor of shape [num_timesteps, C, latent_H, latent_W]
            pred_top_idx: A tensor of shape [1, K]
        """
        raise NotImplementedError


class TTAGradientDescent_Class(TTABase_Class):

    def evaluate(self, batch, before_tta=False, after_tta=False):
        """Evaluate classifier predictions
        """
        # Classify with the classifier
        with torch.no_grad():
            self.class_model.eval()
            image = batch["test_image_disc"]
            logits, pred_topk_idx, pred_class_idx, topk_class_idx = (
                self.classify(image)
            )

        # Compute if classification is correct
        gt_class_idx = batch['class_idx'].squeeze()
        if self.config.input.use_objectnet and self.config.model.use_dit:
            correct = pred_class_idx in gt_class_idx
            correct = torch.tensor(correct).to(pred_class_idx.device)
        else:
            correct = pred_class_idx == gt_class_idx

        # Keep track of the correctness among all images
        if before_tta:
            prefix = 'before_tta'
            self.before_tta_acc.append(correct)
        elif after_tta:
            self.after_tta_acc.append(correct)
            prefix = 'after_tta'
        else:
            prefix = ''

        # Output stats
        stats_dict = {}
        # Keep track of the indices of the top-5 classes, which would be used
        # in `tta_model.gradient_descent_forward``
        if before_tta and self.config.tta.adapt_topk != -1:
            stats_dict['pred_topk_idx'] = pred_topk_idx
        else:
            stats_dict['pred_topk_idx'] = None
        stats_dict[f'{prefix}_correct'] = correct.cpu()
        stats_dict[f'{prefix}_logits'] = logits.cpu()
        stats_dict[f'{prefix}_pred_class_idx'] = pred_class_idx.cpu()
        stats_dict[f'{prefix}_topk_class_idx'] = topk_class_idx.cpu()
        stats_dict[f'{prefix}_topk_idx'] = pred_topk_idx.cpu()

        return stats_dict

    def forward(self, image, x_start=None, t=None, noise=None, pred_top_idx=None):
        """This function compute diffusion loss using current classifier
        predictions.
        """
        # Classify with the classifier
        logits = self.classify(image)[0]
        BS, C = logits.shape[:2]

        # Pick top-K predictions
        if pred_top_idx is not None:
            pred_top_idx = pred_top_idx.squeeze(0)
        else:
            pred_top_idx = torch.arange(C, device=logits.device)

        logits = logits[:, pred_top_idx]
        class_text_embeddings = self.class_text_embeddings[pred_top_idx, :]

        # Compute conditional text embeddings using weighted-summed predictions
        probs = logits.softmax(-1)
        if self.config.model.use_dit:
            probs = probs.unsqueeze(-1)
            class_text_embeddings = (
                class_text_embeddings.unsqueeze(0).repeat(BS, 1, 1)
            )
        else:
            probs = probs[:, :, None, None]
            class_text_embeddings = (
                class_text_embeddings.unsqueeze(0).repeat(BS, 1, 1, 1)
            )

        context = (probs * class_text_embeddings).sum(1)
        # Predict noise with the diffusion model
        pred_noise = self._unet_pred_noise(x_start, t, noise, context)

        # Compute diffusion loss
        if self.config.tta.loss == "l1":
            loss = torch.nn.functional.l1_loss(pred_noise, noise)
        else:
            loss = torch.nn.functional.mse_loss(pred_noise, noise)

        return loss , logits
