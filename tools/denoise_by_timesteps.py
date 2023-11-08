"""Main script for Diffusion-TTA"""
import os
import copy
import random
import warnings

import hydra
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf, open_dict
import numpy as np
import torch
import torch.backends.cudnn as cudnn
torch.backends.cudnn.benchmark = True
import PIL.Image as Image
import ipdb
st = ipdb.set_trace

from dataset.catalog import DatasetCatalog
from diff_tta import utils, engine, vis_utils
from diff_tta.models.tta import TTAGradientDescent_Class
from diff_tta.models import build



class TTAGradientDescent_Class_Vis(TTAGradientDescent_Class):

    def visualize_denoising(self, image, x_start=None, t=None, noise=None, pred_top_idx=None):
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

        cond_context = (probs * class_text_embeddings).sum(1)
        uncond_context = self.unet_model.y_embedder.embedding_table.weight[1000]
        uncond_context = uncond_context.unsqueeze(0).expand(BS, -1)
        context = torch.cat((cond_context, uncond_context), dim=0)

        # Dont use uncondition context for simplicity
        nt = t.shape[0]
        if self.config.model.use_dit:
            x_start = x_start.expand(nt, -1, -1, -1)
            noised_latent = self.diffusion_model.q_sample(
                x_start=x_start, t=t, noise=noise
            )
            model_kwargs = dict(context=context, y=None, cfg_scale=1.0)
            sample = self.diffusion_model.p_sample(
                self.unet_model.forward_with_cfg, noised_latent, t, clip_denoised=False,
                model_kwargs=model_kwargs,
            )
            samples = sample["pred_xstart"]
        else:
            raise NotImplementedError

        return samples, noised_latent


def denoise(batch, tta_model,
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
    latent = engine.prepare_vae_latent(batch, autoencoder, image_renormalizer)
    total_timestep = engine.prepare_total_timesteps(config, tta_model)

    # Initiate timesteps and noise
    timesteps = torch.tensor([222, 111]).long()
    timesteps = timesteps.to(device)
    bs = timesteps.shape[0]

    c, h, w = latent.shape[1:]
    noise = torch.randn((1, c, h, w), device=device)
    noise = noise.expand(bs, -1, -1, -1)
    latent = latent.expand(bs, -1, -1, -1)

    samples, noised_latent = tta_model.visualize_denoising(
        batch["image_gen"], x_start=latent, t=timesteps, noise=noise,
        pred_top_idx=pred_top_idx
    )
    samples = autoencoder.decode(samples / 0.18215).sample
    noised_samples = autoencoder.decode(noised_latent / 0.18215).sample

    samples = torch.cat(
        (samples + 1).mul(127.5).chunk(bs, dim=0), dim=-1
    )
    samples = samples.squeeze(0).permute(1, 2, 0)
    samples = samples.data.cpu().numpy().astype(np.uint8)
    samples = Image.fromarray(samples, mode='RGB')

    noised_samples = torch.cat(
        (noised_samples + 1).mul(127.5).chunk(bs, dim=0), dim=-1
    )
    noised_samples = noised_samples.squeeze(0).permute(1, 2, 0)
    noised_samples = noised_samples.data.cpu().numpy().astype(np.uint8)
    noised_samples = Image.fromarray(noised_samples, mode='RGB')

    return samples, noised_samples


def tta_one_epoch(config, dataloader, tta_model, optimizer, scaler,
                  autoencoder, image_renormalizer):
    """Perform test time adaptation over the entire dataset.

    Args:
        config: configuration object for hyper-parameters.
        dataloader: The dataloader for the dataset.
        tta_model: A test-time adaptation wrapper model.
        optimizer: A gradient-descent optimizer for updating classifier.
        scaler: A gradient scaler used jointly with optimizer.
        autoencoder: A pre-trained autoencoder model (e.g. VQVAE).
        image_renormalizer: An object for renormalizing images.
    """
    
    cwd = config.cwd
    
    tta_model.eval()

    # Keep a copy of the original model state dict, so that we can reset the
    # model after each image
    tta_class_state_dict = copy.deepcopy(tta_model.state_dict())
    
    # Enlarge batch size by accumulating gradients over multiple iterations
    config.tta.gradient_descent.train_steps = (
        config.tta.gradient_descent.train_steps
        * config.tta.gradient_descent.accum_iter
    )

    # Start iterations
    #for img_ind in range(150):
    for img_ind in [7]:

        # Fetch data from the dataset
        batch = dataloader.dataset[img_ind]
        batch = engine.preprocess_input(batch, config.gpu)
    
        # We will classify before and after test-time adaptation via
        # gradient descent. We run tta_model.evaluate(batch, after_tta=True) to
        # save the classification results

        # Step 1: Predict pre-TTA classification. The results are saved in
        # `before_tta_stats_dict` and `tta_model.before_tta_acc`
        before_tta_stats_dict = tta_model.evaluate(batch, before_tta=True)
        pred_topk_idx = before_tta_stats_dict['pred_topk_idx'][0].cpu()

        # Visualize denoising
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)
        denoised_samples, noised_samples = denoise(
            batch, tta_model, autoencoder, image_renormalizer,
            config, pred_topk_idx
        )
        samples = np.concatenate((np.array(noised_samples), np.array(denoised_samples)), axis=0)
        Image.fromarray(samples, mode='RGB').save('debug_0.png')

        for tta_step in range(1, 6):
            np.random.seed(0)
            random.seed(0)
            torch.manual_seed(0)
            # Step 2: TTA by gradient descent
            _ = engine.tta_one_image_by_gradient_descent(
                batch, tta_model, optimizer, scaler,
                autoencoder, image_renormalizer, config,
                before_tta_stats_dict['pred_topk_idx']
            )

            np.random.seed(0)
            random.seed(0)
            torch.manual_seed(0)
            denoised_samples, noised_samples = denoise(
                batch, tta_model, autoencoder, image_renormalizer,
                config, pred_topk_idx
            )
            samples = np.concatenate((np.array(noised_samples), np.array(denoised_samples)), axis=0)
            Image.fromarray(samples, mode='RGB').save(f'debug_{tta_step:d}.png')

        # Reload the original model state dict
        tta_model.load_state_dict(tta_class_state_dict)
        optimizer = build.load_optimizer(config, tta_model)


def get_dataset(config):
    """Instantiate the dataset object."""
    Catalog = DatasetCatalog(config)

    dataset_dict = getattr(Catalog, config.input.dataset_name)

    target = dataset_dict['target']
    params = dataset_dict['train_params']
    if config.input.dataset_name == "ObjectNetSubsetNew":
        params.update({'use_dit': config.model.use_dit})
    dataset = utils.instantiate_from_config(dict(target=target, params=params))

    return dataset


@hydra.main(config_path="../diff_tta/config", config_name="config")
def run(config):
    with open_dict(config):
        config.log_dir = os.getcwd()
        print(f"Logging files in {config.log_dir}")
        config.cwd = get_original_cwd()
        config.gpu = None if config.gpu < 0 else config.gpu

    # Hydra automatically changes the working directory, but we stay at the
    # project directory.
    os.chdir(config.cwd)

    print(OmegaConf.to_yaml(config))
    
    if config.input.dataset_name == "ObjectNetSubsetNew":
        config.input.use_objectnet = True

    if config.seed is not None:
        np.random.seed(config.seed)
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    run_worker(config)


def run_worker(config):

    if config.gpu is not None:
        print("Use GPU: {} for training".format(config.gpu))

    print("=> Loading dataset")
    dataset = get_dataset(config)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=True,
        sampler=None,
        drop_last=True
    )
    
    # create model
    print("=> Creating model ")
    model, autoencoder, image_renormalizer = (
        build.create_models(config, dataset.classes, dataset.class_names)
    )
    optimizer = build.load_optimizer(config, model)
    scaler = torch.cuda.amp.GradScaler()

    tta_one_epoch(config, dataloader, model, optimizer, scaler,
                  autoencoder, image_renormalizer)


if __name__ == '__main__':
    run()


