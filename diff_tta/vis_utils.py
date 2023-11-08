"""Utility functions for visualization."""
import torch
from PIL import Image
import numpy as np
import wandb
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from diff_tta import utils



def visualize_classification_with_image(batch, config, dataset,
                                        logits, topk_idx,
                                        pred_class_idx, topk_class_idx,
                                        wandb_dict):
    """A wrapper to visualize classification top-K probability along with image

    Args:
        batch: A dictionary with following entries:
                - image_gen: a tensor of shape (1, 3, H, W)
                - image_disc: a tensor of shape (1, 3, H, W)
                - test_image_gen: a tensor of shape (1, 3, H, W)
                - test_image_disc: a tensor of shape (1, 3, H, W)
        config: A `config` object
        dataset: A `dataset` object
        logits: A tensor of shape (1, num_classes)
        topk_idx: A tensor of shape (1, K)
        pred_class_idx: A tensor of shape (1, 1)
        topk_class_idx: A tensor of shape (1, K)
        wandb_dict: A dictionary to save the visualization results for wandb
    
    Returns:
        wandb_dict: update in-place
    """
    image_disc_pil = unorm_image_to_pil(batch["image_disc"][:1],
                                        config.input.mean, config.input.std)
    image_gen_pil = unorm_image_to_pil(batch["image_gen"][:1],
                                       config.input.mean, config.input.std)
    test_image_disc_pil = unorm_image_to_pil(batch["test_image_disc"][:1],
                                             config.input.mean, config.input.std)
    test_image_gen_pil = unorm_image_to_pil(batch["test_image_gen"][:1],
                                            config.input.mean, config.input.std)
    
    # visualize ground-truth
    if config.input.use_objectnet and config.model.use_dit:
        class_idx = [int(idx) for idx in batch['class_idx'][0]]
        class_name = [
            dataset.imagenet_class_index_mapping[str(idx)][1] for idx in class_idx
        ]
        class_name = "/".join(class_name)
    else:
        class_idx = int(batch['class_idx'][0])
        class_name = dataset.class_names[class_idx]

    pred_class_idx = int(pred_class_idx.squeeze(0).item())
    topk_class_idx = [
        int(l) for l in topk_class_idx.squeeze(0).tolist()
    ]
    topk_probs = torch.gather(logits, 1, topk_idx).softmax(-1).squeeze(0)
    if config.input.use_objectnet and config.model.use_dit:
        pred_class_name = (
            dataset.imagenet_class_index_mapping[str(pred_class_idx)]
        )
    else:
        pred_class_name = dataset.class_names[pred_class_idx]
        pred_all_topk_names = [
            dataset.class_names[int(i)] for i in topk_class_idx
        ]
        K = topk_probs.shape[0]
        output_string = ""
        for i, name_val in enumerate(pred_all_topk_names):
            output_string += f"{name_val}: {topk_probs[i]:.2f}, "
        toppred_table = wandb.Table(
            columns=[f"top{K:d}"], data=[[f"{output_string}"]]
        )
        
        print(f"GT: {class_name}")
        print(f"top pred: {output_string}")
        wandb_dict['topk_pred_table'] = toppred_table

    wandb_dict['input_image_disc'] = wandb.Image(
        np.array(image_disc_pil),
        caption=f"GT: {class_name}, Pred: {pred_class_name}"
    )
    wandb_dict['input_image_gen'] = wandb.Image(
        np.array(image_gen_pil),
        caption=f"GT: {class_name}, Pred: {pred_class_name}"
    )
    wandb_dict['test_input_image_disc'] = wandb.Image(
        np.array(test_image_disc_pil),
        caption=f"GT: {class_name}, Pred: {pred_class_name}"
    )
    wandb_dict['test_input_image_gen'] = wandb.Image(
        np.array(test_image_gen_pil),
        caption=f"GT: {class_name}, Pred: {pred_class_name}"
    )

    return wandb_dict


def visualize_diffusion_loss(diffusion_loss, config, wandb_dict):
    """Plot diffusion loss curve over TTA steps.

    Returns:
        wandb_dict: update in-place
    """
    diffusion_loss = (
        np.array(diffusion_loss)
            .reshape(-1, config.tta.gradient_descent.accum_iter)
    )
    diffusion_loss = diffusion_loss.mean(-1)
    diffusion_curve = plot_tta_curve(
        {"diffusion loss": diffusion_loss},
    )
    wandb_dict["diffusion loss over tta"] = wandb.Image(diffusion_curve)                

    return wandb_dict


def visualize_classification_improvements(before_tta_acc,
                                          after_tta_acc,
                                          before_tta_correct,
                                          after_tta_correct,
                                          wandb_dict):
    """Plot per-image improvements before and after TTA.

    Returns:
        wandb_dict: update in-place
    """

    before_avg_acc = sum(before_tta_acc) / len(before_tta_acc)
    after_avg_acc = sum(after_tta_acc) / len(after_tta_acc)
    wandb_dict["before_avg_acc"] = before_avg_acc
    wandb_dict["after_avg_acc"] = after_avg_acc                
    wandb_dict["improvement_avg"] = (after_avg_acc - before_avg_acc)*100
    print("Before-TTA avg acc: {:.2f}".format(before_avg_acc.item()),
          "After-TTA avg acc: {:.2f}".format(after_avg_acc.item()))
    
    per_image_improvement = after_tta_correct - before_tta_correct
    wandb_dict["per_image_improvement"] = per_image_improvement
    return wandb_dict


def unorm_image_to_pil(image_val, mean, std):
    """Unformalize image tensor and convert to PIL image
    """
    image_unnorm = utils.UnNormalize(
        mean=mean, std=std
    )(image_val).cpu()
    image_val = Image.fromarray(
        (image_unnorm * 255).to(torch.uint8).permute(0,2,3,1).squeeze(0).numpy()
    )
    return image_val


def plot_tta_curve(ys, tta_steps=None):
    """A helper function to plot curve over TTA steps
    Args:
        ys: a dictionary {name: values}
    """
    vis_data = None
    for k, v in ys.items():
        if isinstance(v, (list, tuple)):
            v = np.array(v)
        elif isinstance(v, torch.Tensor):
            v = v.data.cpu().numpy()
        ys[k] = v
    
    if tta_steps is None:
        tta_steps = np.arange(list(ys.values())[0].shape[0])
    elif isinstance(tta_steps, (list, tuple)):
        tta_steps = np.array(tta_steps)
    elif isinstance(tta_steps, torch.Tensor):
        tta_steps = tta_steps.data.cpu().numpy()

    fig = plt.figure(figsize=(10, 10))
    for name, y in ys.items():
        plt.plot(tta_steps, y, label="{:s}".format(name))            

    plt.legend()
    plt.ylabel('curve over TTA steps')
    plt.xlabel('time steps')
    fig.canvas.draw()
    vis_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    vis_data = vis_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    return vis_data