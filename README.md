<div align="center">

<!-- TITLE -->
# **Diffusion-TTA: Test-time Adaptation of Discriminative Models via Generative Feedback**

 [![arXiv](https://img.shields.io/badge/cs.LG-arXiv:2311.16102-b31b1b.svg)](https://arxiv.org/abs/2311.16102)
[![Website](https://img.shields.io/badge/🌎-Website-blue.svg)](https://diffusion-tta.github.io)
</div>

This is the official implementation of the paper [Diffusion-TTA: Test-time Adaptation of Discriminative Models via Generative Feedback](https://diffusion-tta.github.io/) by Mihir Prabhudesai, Tsung-Wei Ke, Alexander Li, Deepak Pathak, and Katerina Fragkiadaki.
<!-- DESCRIPTION -->

## Abstract

The advancements in generative modeling, particularly the advent of diffusion models, have sparked a fundamental question: how can these models be effectively used for discriminative tasks? In this work, we find that generative models can be great test-time adapters for discriminative models. Our method, Diffusion-TTA, adapts pre-trained discriminative models such as image classifiers, segmenters and depth predictors, to each unlabelled example in the test set using generative feedback from a diffusion model.

We achieve this by modulating the conditioning of the diffusion model using the output of the discriminative model. We then maximize the image likelihood objective by backpropagating the gradients to discriminative model’s parameters. We show Diffusion-TTA significantly enhances the accuracy of various large-scale pre-trained discriminative models, such as, ImageNet classifiers, CLIP models, image pixel labellers and image depth predictors. Diffusion-TTA outperforms existing test-time adaptation methods, including TTT-MAE and TENT, and particularly shines in online adaptation setups, where the discriminative model is continually adapted to each example in the test set.


## Diffusion-TTA

**Generative diffusion models are great test-time adapters for discriminative models.** Our method consists of discriminative and generative modules. Given an image $x$, the discriminative model $f_{\theta}$ predicts task output $y$. The task output $y$ is transformed into condition $c$. Finally, we use the generative diffusion model $\epsilon_{\phi}$ to measure the likelihood of the input image, conditioned on $c$. This consists of using the diffusion model $\epsilon_{\phi}$ to predict the added noise $\epsilon$ from the noisy image $x_t$ and condition $c$. We maximize the image likelihood using the diffusion loss by updating the discriminative and generative model weights via backpropagation. 


![alt text](figures/arch.gif)

Our model improves classification test performance without the need of ground-truth labels. Classification errorr are corrected by minimizing the diffusion loss.

![alt text](figures/tta_3.gif)

## Features
- [x] Adaptation of ImageNet-trained classifiers with DiT
- [ ] Adaptation of CLIP models with Stable Diffusion

## Installation 
Create a conda environment with the following command:
```bash
conda update conda
conda env create -f environment.yml
conda activate diff_tta
```

### Prepare DiT
Clone our DiT branch forked from the official repo.  We adapt the original code so that we vary the class text embeddings.  We modify the DiT code base to enable conditioning of class text embeddings weighted average with predicted probabilities.
```
git clone https://github.com/mihirp1998/DiT.git
mv DiT diff_tta/models
mkdir pretrained_models
```

## Prepare Datasets

By default, we expect all datasets put under the local `data/` directory.  You can set [`input.root_path`](https://github.com/mihirp1998/Diffusion-TTA/blob/3c1eda48d31c42f08cb2d75da36e8d18077ec7e0/diff_tta/config/config.yaml#L46) to your local data directory.
```
# By default, our code base expect
./data/
   |-------- imagenet-a/
   |-------- imagenet-r/
   |-------- ImageNet/val/
   |-------- ImageNet-C/gaussian_noise/5
   |-------- imagenetv2-matched-frequency-format-val/
   |-------- imagenet-styletransfer-v2/val
```

We provide a bashscript to download ImageNet-A and ImageNet-R
```
bash download_dataset.sh
```

For ImageNet-v2, the testing set is hosted on [HuggingFace](https://huggingface.co/datasets/vaishaal/ImageNetV2/tree/main).  Please download `imagenetv2-matched-frequency.tar.gz` and extract to `./data/`.

For ImageNet-C, please follow the authors' [instruction](https://github.com/hendrycks/robustness) to download and extract the dataset.

For Stylized-ImageNet, we provide the rendered [validation set](https://drive.google.com/drive/folders/1TFCBRkA8r5ik7uxIYIXaUbGbo1glUl5h?usp=drive_link).

## Commands to Get Started
Our classification results vary with the randomly sampled noises and timesteps during TTA.  To reproduce our results, we provide the commands used in each experiment. See [Getting_Startted.md](./GETTING_STARTED.md) for details.


## Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{prabhudesai2023difftta,
      title={Test-time Adaptation of Discriminative Models via Diffusion Generative Feedback},
      author={Prabhudesai, Mihir and Ke, Tsung-Wei and Li, Alexander C. and Pathak, Deepak and Fragkiadaki, Katerina},
      year={2023},
      booktitle={Conference on Neural Information Processing Systems},
}
```

## License
This code base is released under the MIT License (refer to the LICENSE file for details).
