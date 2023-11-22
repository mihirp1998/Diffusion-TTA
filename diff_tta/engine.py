"""Define training/inference logic for TTA"""
import torch
from tqdm import tqdm

import diff_tta.utils as utils


def preprocess_input(batch, device):
    """A helper function to put data onto GPU.
    """
    new_batch = {}
    if device is None:
        device = 'cuda'
    for k, v in batch.items():
        if isinstance(v, str):
            pass
        else:
            v = torch.tensor(v).to(device, non_blocking=True).unsqueeze(0)
        new_batch[k] = v
    return new_batch


@torch.no_grad()
def prepare_vae_latent(batch, autoencoder, image_renormalizer):
    """A helper function to prepare latent code from VQVAE
    Args:
        batch: a dictionary with following entries:
               - image_gen: a tensor of shape (1, 3, H, W)
               - image_disc: a tensor of shape (1, 3, H, W)
        autoencoder: a autoencoder that encodes images to latent space (VQVAE)
        image_renoramlizer: a function that unnorm and renorm dataset processed
                            images in consistent with the normalization used
                            by the autoencoder

    Returns:
        latent: a tensor of shape (1, C, H, W)
    """
    # Renormalize the image to be consistent with the autoencoder
    renormed_image = image_renormalizer(batch['image_gen']).detach()
    x0 = autoencoder.encode(renormed_image).latent_dist.mean
    latent = x0 *  0.18215
    return latent


@torch.no_grad()
def prepare_total_timesteps(config, tta_model):
    """A helper function to infer total timesteps of diffusion model
    Args:
        config: A config object
        tta_model: A nn.Module object that contains diffusion model

    Returns:
        total_timestep: A scalar
    """

    if config.model.use_dit:
        if config.model.override_total_steps != -1:
            total_timestep = config.model.override_total_steps
        else:
            total_timestep = tta_model.diffusion_timesteps()
    else:
        total_timestep = tta_model.scheduler.num_train_timesteps

    return total_timestep


def tta_one_image_by_gradient_descent(batch, tta_model, optimizer, scaler,
                                      autoencoder, image_renormalizer,
                                      config, pred_top_idx):
    """TTA by gradient descent

    Args:
        batch: a dictionary with following entries:
               - image_gen: a tensor of shape (1, 3, H, W)
               - image_disc: a tensor of shape (1, 3, H, W)
        tta_model: a TTA model that adapts classifiers and predicts classification
        optimizer: a SGD/Adam optimizer
        scaler: a GradScaler object
        autoencoder: a autoencoder that encodes images to latent space (VQVAE)
        image_renoramlizer: a function that unnorm and renorm dataset processed
                            images in consistent with the normalization used
                            by the autoencoder
        config: a config object
        pred_top_idx: a list of top-K indices of the predicted classes

    Returns:
        losses: a list of losses
        all_preds: a list of predictions
    """
    device = batch["image_disc"].device

    # Prepare the latent code and diffusion model
    latent = prepare_vae_latent(batch, autoencoder, image_renormalizer)
    total_timestep = prepare_total_timesteps(config, tta_model)

    # Perform adaptation
    losses = []
    all_preds = []
    for step in tqdm(range(config.tta.gradient_descent.train_steps)):

        # Initiate timesteps and noise
        bs = config.input.batch_size
        timesteps = utils.initiate_time_steps(step, total_timestep, bs, config)
        timesteps = timesteps.to(device)

        c, h, w = latent.shape[1:]
        if not config.tta.use_same_noise_among_timesteps:
            noise = torch.randn((bs, c, h, w), device=device)
        else:
            noise = torch.randn((1, c, h, w), device=device)
            noise = noise.repeat(bs, 1, 1, 1)

        # Model adaptation
        loss, preds = tta_model(
            image=batch["image_disc"],
            x_start=latent,
            t=timesteps,
            noise=noise,
            pred_top_idx=pred_top_idx
        )
        all_preds.append(preds)

        loss_vis = loss.item()
        losses.append(loss_vis)
        loss = loss / config.tta.gradient_descent.accum_iter

        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        if ((step + 1) % config.tta.gradient_descent.accum_iter == 0):
            scaler.step(optimizer)
            optimizer.zero_grad()
            scaler.update()

    return losses, all_preds
