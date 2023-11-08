import torch
import torch.nn as nn
import torchvision
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
    StableDiffusionPipeline,
    EulerDiscreteScheduler
)
from transformers import CLIPTextModel, CLIPTokenizer

from diff_tta.utils import get_obj_from_str
from diff_tta.models.DiT.models import DiT_XL_2
from diff_tta.models.DiT.download import find_model
from diff_tta.models.clip_classifier import ClipClassifier
from diff_tta.models.DiT.diffusion import create_diffusion
from diff_tta import utils



def load_dit_model(config, device):
    """Load DiT model"""
    #@param ["stabilityai/sd-vae-ft-mse", "stabilityai/sd-vae-ft-ema"]
    vae_model = "stabilityai/sd-vae-ft-ema"
    image_size = config.input.sd_img_res
    latent_size = int(image_size) // 8

    model = DiT_XL_2(input_size=latent_size).to(device)
    state_dict = find_model(f"DiT-XL-2-{image_size}x{image_size}.pt")
    model.load_state_dict(state_dict)
    model.eval() # important!
    vae = AutoencoderKL.from_pretrained(vae_model).to(device)
    vae.eval()
    # default: 1000 steps, linear noise schedule
    diffusion = create_diffusion(timestep_respacing="")
    image_renormalizer = utils.VQVAEUnNormalize(
        mean=config.input.mean, std=config.input.std
    )

    if config.model.adapt_only_classifier:
        for m in [vae, model]:
            for param in m.parameters():
                param.requires_grad = False

    if config.model.freeze_vae:
        for m in [vae]:
            for param in m.parameters():
                param.requires_grad = False

    return vae, model, diffusion, image_renormalizer


def load_sd_model(config, device, classes):
    """Load Stable Diffusion model"""
    dtype = torch.float32

    image_renormalizer = utils.VQVAEUnNormalize(
        mean=config.input.mean, std=config.input.std
    )

    if config.model.sd_version == '1-4':
        if config.model.use_flash:
            model_id = "CompVis/stable-diffusion-v1-4"
            scheduler = EulerDiscreteScheduler.from_pretrained(
                model_id, subfolder="scheduler"
            )
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id, scheduler=scheduler, torch_dtype=dtype
            ).to(device)
            pipe.enable_xformers_memory_efficient_attention()
            vae = pipe.vae.to(device)
            tokenizer = pipe.tokenizer
            text_encoder = pipe.text_encoder.to(device)
            unet = pipe.unet.to(device)
        else:
            vae = AutoencoderKL.from_pretrained(
                f"CompVis/stable-diffusion-v{config.model.sd_version}",
                subfolder="vae", torch_dtype=dtype
            ).to(device)
            tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
            text_encoder = CLIPTextModel.from_pretrained(
                "openai/clip-vit-large-patch14", torch_dtype=dtype
            ).to(device)
            unet = UNet2DConditionModel.from_pretrained(
                f"CompVis/stable-diffusion-v{config.model.sd_version}",
                subfolder="unet", torch_dtype=dtype
            ).to(device)
            scheduler_config = get_scheduler_config(config)
            scheduler = DDPMScheduler(
                num_train_timesteps=scheduler_config['num_train_timesteps'],
                beta_start=scheduler_config['beta_start'],
                beta_end=scheduler_config['beta_end'],
                beta_schedule=scheduler_config['beta_schedule']
            )
    elif config.model.sd_version == '2-1':
        model_id = "stabilityai/stable-diffusion-2-1-base"
        scheduler = EulerDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, scheduler=scheduler, torch_dtype=dtype
        ).to(device)
        pipe.enable_xformers_memory_efficient_attention()
        vae = pipe.vae.to(device)
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder.to(device)
        unet = pipe.unet.to(device)
    else:
        raise NotImplementedError

    class_text_embeddings = utils.prepare_class_text_embeddings(
        device, tokenizer, text_encoder, class_names=classes
    )
    class_text_embeddings = class_text_embeddings.detach()

    if config.model.adapt_only_classifier:
        for m in [vae, text_encoder, unet]:
            for param in m.parameters():
                param.requires_grad = False
    for m in [vae, text_encoder]:
        for param in m.parameters():
            param.requires_grad = False

    return (vae, tokenizer, text_encoder, unet, scheduler,
            image_renormalizer, class_text_embeddings)


def get_scheduler_config(config):
    assert config.model.sd_version in {'1-4', '2-1'}
    if config.model.sd_version == '1-4':
        schedule_config = {
            "_class_name": "PNDMScheduler",
            "_diffusers_version": "0.7.0.dev0",
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "num_train_timesteps": 1000,
            "set_alpha_to_one": False,
            "skip_prk_steps": True,
            "steps_offset": 1,
            "trained_betas": None,
            "clip_sample": False
        }
    elif config.model.sd_version == '2-1':
        schedule_config = {
            "_class_name": "EulerDiscreteScheduler",
            "_diffusers_version": "0.10.2",
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "clip_sample": False,
            "num_train_timesteps": 1000,
            "prediction_type": "epsilon",
            "set_alpha_to_one": False,
            "skip_prk_steps": True,
            "steps_offset": 1,  # todo
            "trained_betas": None
        }
    else:
        raise NotImplementedError

    return schedule_config


def get_class_model(config, classes):
    """Load classification model"""
    if "clip" in config.model.class_arch:
        class_model = ClipClassifier(classes, config.model.class_arch)
        class_model.to(torch.float32)
    else:
        class_model = (
            torchvision.models.__dict__[config.model.class_arch](pretrained=True)
        )
    return class_model


def create_models(config, classes, zs_classes = None):
    """Create a wrapper model for TTA"""
    device = "cuda" if config.gpu is None else "cuda:{}".format(config.gpu)
    if config.model.use_dit:
        vae, unet, diffusion, image_renormalizer = load_dit_model(config, device)
        tokenizer = None
        text_encoder = None
        class_text_embeddings = None
        scheduler = None
    else:
        (vae, tokenizer, text_encoder, unet, scheduler,
         image_renormalizer, class_text_embeddings) = (
            load_sd_model(config, device, zs_classes)
        )
        diffusion = None
        text_encoder.eval()

    vae.eval()
    unet.eval()

    class_model = get_class_model(config ,zs_classes)
    class_model.eval()

    tta_model = get_obj_from_str(config.tta.model)(unet_model=unet,
                                 class_model=class_model,
                                 scheduler=scheduler,
                                 diffusion=diffusion,
                                 class_text_embeddings=class_text_embeddings,
                                 classes=classes,
                                 config=config)
    tta_model.eval()

    tta_model.to(device)
    vae = vae.to(device)

    return tta_model, vae, image_renormalizer


def load_optimizer(config, model):
    """Reset momentum gradients in the optimizer"""
    params = model.parameters()
    if config.model.freeze_class_embeds:
        model_layers, model_names  = get_children("model",model)

        if ("resnet" in config.model.class_arch
            or "vit" in config.model.class_arch
            or 'convnext' in config.model.class_arch):
            params = []
            for layer in model_layers[:-1]:
                params.append({'params': layer.parameters()})
        elif "clip" in config.model.class_arch:
            index_val = model_names.index('final_fc')
            model_layers.pop(index_val)
            params = []

            for layer in model_layers:
                params.append({'params': layer.parameters()})
        else:
            raise NotImplementedError

    if config.tta.gradient_descent.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params, lr=config.tta.gradient_descent.base_learning_rate,
            weight_decay=config.tta.gradient_descent.weight_decay,
            momentum=config.tta.gradient_descent.optimizer_momentum
        )
    else:
        optimizer = torch.optim.AdamW(
            params, lr=config.tta.gradient_descent.base_learning_rate,
            weight_decay=config.tta.gradient_descent.weight_decay
        )
    optimizer.zero_grad()
    return optimizer


def get_children(name, model: nn.Module):
    # get children form model!
    # children = list(model.children())
    children = []
    names = []
    for n,l in model.named_children():
        children.append(l)
        names.append(n)

    flatt_children = []
    flatt_names = []
    if children == []:
        # if model has no children; model is last child! :O
        return model,name
    else:
       # look for children from children... to the last child!
       for name, child in zip(names, children):
            try:
                flat_child, flat_name = get_children(name, child)
                flatt_children.extend(flat_child)
                flatt_names.extend(flat_name)
            except TypeError:
                flat_child, flat_name = get_children(name, child)
                flatt_children.append(flat_child)
                flatt_names.append(flat_name)
    return flatt_children, flatt_names
