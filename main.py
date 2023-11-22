"""Main script for Diffusion-TTA"""
import os
import copy
import random
import warnings

import wandb
import hydra
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf, open_dict
from mergedeep import merge
import numpy as np
import pickle
import torch
import torch.backends.cudnn as cudnn
torch.backends.cudnn.benchmark = True

from dataset.catalog import DatasetCatalog
from diff_tta import utils, engine
from diff_tta.vis_utils import (
    visualize_classification_with_image,
    visualize_diffusion_loss,
    visualize_classification_improvements,
)
from diff_tta.models import build


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
    discrete_sampling_accuracy = []
    
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
    start_index = 0
    last_index = len(dataloader.dataset)
    for img_ind in range(start_index, last_index):
        # Enable/disable to upload visualization to wandb
        visualize = (
            (config.log_freq > 0 and img_ind % config.log_freq == 0)
            or img_ind == last_index - 1
        )

        # The dictionary for visualization
        wandb_dict = {}

        # Fetch data from the dataset
        print(f"\n\n Example: {img_ind}/{last_index} \n\n")
        batch = dataloader.dataset[img_ind]
        batch = engine.preprocess_input(batch, config.gpu)
    
        # We will classify before and after test-time adaptation via
        # gradient descent. We run tta_model.evaluate(batch, after_tta=True) to
        # save the classification results

        # Step 1: Predict pre-TTA classification. The results are saved in
        # `before_tta_stats_dict` and `tta_model.before_tta_acc`
        before_tta_stats_dict = tta_model.evaluate(batch, before_tta=True)

        # Step 2: TTA by gradient descent
        losses, after_tta_outputs = engine.tta_one_image_by_gradient_descent(
            batch, tta_model, optimizer, scaler,
            autoencoder, image_renormalizer, config,
            before_tta_stats_dict['pred_topk_idx']
        )

        # Step 3: Predict post-TTA classification. The results are saved in
        # `after_tta_stats_dict` and `tta_model.after_tta_acc`
        after_tta_stats_dict = tta_model.evaluate(batch, after_tta=True)

        # Reload the original model state dict
        if not config.tta.online:
            tta_model.load_state_dict(tta_class_state_dict)
            optimizer = build.load_optimizer(config, tta_model)

        if visualize:
            # wandb_dict is updated in-place
            wandb_dict = visualize_classification_with_image(
                batch, config, dataloader.dataset,
                before_tta_stats_dict["before_tta_logits"],
                before_tta_stats_dict["before_tta_topk_idx"],
                before_tta_stats_dict["before_tta_pred_class_idx"],
                before_tta_stats_dict["before_tta_topk_class_idx"],
                wandb_dict
            )

            wandb_dict = visualize_diffusion_loss(losses, config, wandb_dict)

        # Plot accuracy curve every image
        wandb_dict = visualize_classification_improvements(
            tta_model.before_tta_acc, tta_model.after_tta_acc,
            before_tta_stats_dict["before_tta_correct"].float(),
            after_tta_stats_dict["after_tta_correct"].float(),
            wandb_dict
        )

        # Save the results to the disck
        wandb_run_name = wandb.run.name
        stats_folder_name = f'stats/{wandb_run_name}/'
        os.makedirs(stats_folder_name, exist_ok=True)
        
        if config.save_results:
            stats_dict = {}
            stats_dict['accum_iter'] = config.tta.gradient_descent.accum_iter
            stats_dict['filename'] = batch['filepath']
            stats_dict['losses'] = losses
            stats_dict['gt_idx'] = batch['class_idx'][0]
            stats_dict = merge(stats_dict, before_tta_stats_dict, after_tta_stats_dict)
            file_index = int(batch['index'].squeeze())
            store_filename = f"{stats_folder_name}/{file_index:06d}.p"
            pickle.dump(stats_dict, open(store_filename, 'wb'))

        wandb.log(wandb_dict, step=img_ind)


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


@hydra.main(config_path="diff_tta/config", config_name="config")
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
    
    if config.input.dataset_name == "ObjectNetDataset":
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

    wandb.init(project=config.wandb.project, config=config, mode=config.wandb.mode)

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
