"""Main script for Diffusion-TTA"""
import os
import copy
import random
import warnings

import hydra
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf, open_dict
from mergedeep import merge
import numpy as np
import pickle
import torch
import torch.backends.cudnn as cudnn
torch.backends.cudnn.benchmark = True
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import ipdb
st = ipdb.set_trace

from dataset.catalog import DatasetCatalog
from diff_tta import utils, engine, vis_utils
from diff_tta.models import build



def fetch_vis_info(stats_dict, dataset, gt_class_idx, pred_topk_idx,
                   tta_step, out_dict):
    prefix = 'before_tta' if tta_step == 0 else f'after_tta'

    logits = stats_dict[f'{prefix}_logits'][0]
    topk_probs = logits[pred_topk_idx].softmax(-1)

    if tta_step == 0:
        gt_class_name = dataset.class_names[gt_class_idx]
        out_dict['gt_class_name'] = gt_class_name

        topk_class_idx = stats_dict[f'{prefix}_topk_class_idx'][0]
        topk_idx = stats_dict[f'{prefix}_topk_idx'][0]
        assert (pred_topk_idx == topk_idx).all()
        pred_all_topk_names = [
            dataset.class_names[int(i)] for i in topk_class_idx
        ]
        out_dict['pred_topk_class_name'] = pred_all_topk_names

        # cross entropy loss
        gt_idx = (topk_class_idx.cpu() == gt_class_idx.cpu())
        out_dict['gt_idx'] = gt_idx
        ce_loss = -topk_probs[gt_idx].log()
        out_dict['ce_loss'] = [ce_loss]

        out_dict['pred_topk_probs'] = [topk_probs]
    else:
        gt_idx = out_dict['gt_idx']
        ce_loss = -topk_probs[gt_idx].log()
        out_dict['ce_loss'].append(ce_loss)

        out_dict['pred_topk_probs'].append(topk_probs)

    return out_dict


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
    for img_ind in range(150):
        # The dictionary for visualization
        vis_dict = {}

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
        vis_dict = fetch_vis_info(before_tta_stats_dict, dataloader.dataset,
                                  batch["class_idx"], pred_topk_idx, 0, vis_dict)
        if not vis_dict['gt_class_name'] in vis_dict['pred_topk_class_name'][1:]:
            continue

        diff_losses = []
        for tta_step in range(1, 6):
            np.random.seed(0)
            random.seed(0)
            torch.manual_seed(0)
            # Step 2: TTA by gradient descent
            loss, _ = engine.tta_one_image_by_gradient_descent(
                batch, tta_model, optimizer, scaler,
                autoencoder, image_renormalizer, config,
                before_tta_stats_dict['pred_topk_idx']
            )
            loss = sum(loss) / len(loss)
            diff_losses.append(loss)

            # Step 3: Predict post-TTA classification. The results are saved in
            # `after_tta_stats_dict` and `tta_model.after_tta_acc`
            after_tta_stats_dict = tta_model.evaluate(batch, after_tta=True)

            vis_dict = fetch_vis_info(after_tta_stats_dict, dataloader.dataset,
                                      batch["class_idx"], pred_topk_idx,
                                      tta_step, vis_dict)
 

        before_correct = before_tta_stats_dict["before_tta_correct"].float()
        after_correct = after_tta_stats_dict["after_tta_correct"].float()
        improved = (after_correct > before_correct).all()
        decrease_ce = (torch.cat(vis_dict['ce_loss'][:-1]) > torch.cat(vis_dict['ce_loss'][1:])).all()
        if not (improved and decrease_ce):
            continue
        # Reload the original model state dict
        tta_model.load_state_dict(tta_class_state_dict)
        optimizer = build.load_optimizer(config, tta_model)

        # Draw cross entropy loss curve
        image_disc_pil = vis_utils.unorm_image_to_pil(
            batch["image_disc"][:1], config.input.mean, config.input.std
        )

        print('gt_class_name', vis_dict['gt_class_name'])
        print('pred_topk_class_name', vis_dict['pred_topk_class_name'])
        print('ce_losses', vis_dict['ce_loss'])
        print('diff_losses', diff_losses)
        for i in range(6):
            print('topk_probs', vis_dict['pred_topk_probs'][i])
        
        image_disc_pil.save('debug.png')

        st()


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

