"""Define datasets and their parameters."""
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


class DatasetCatalog:
    def __init__(self, config):
        ########### Define image transformations ###########
        mean = config.input.mean
        std = config.input.std
        
        interpolation = InterpolationMode.BILINEAR
        self.test_classification_transforms = T.Compose(
            [
                T.Resize(config.input.disc_img_resize, interpolation=interpolation),
                T.CenterCrop(config.input.disc_img_crop),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
            ]
        )
        self.test_diffusion_transforms = T.Compose(
            [
                T.Resize(config.input.disc_img_resize, interpolation=interpolation),
                T.CenterCrop(config.input.disc_img_crop),
                T.Resize(config.input.sd_img_res, interpolation=interpolation),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
            ]
        )

        self.classification_transforms = self.test_classification_transforms
        self.diffusion_transforms = self.test_diffusion_transforms

        ########### Define datasets ###########
        self.Food101Dataset = {   
            "target": "dataset.dataset_class_label.Food101Dataset",
            "train_params":dict(
                root=config.input.root_path,
                classification_transform=self.classification_transforms,
                diffusion_transform=self.diffusion_transforms,
                test_classification_transform=self.test_classification_transforms,
                test_diffusion_transform=self.test_diffusion_transforms,
                subsample=config.input.subsample,
            ),
        }

        self.Flowers102Dataset = {   
            "target": "dataset.dataset_class_label.Flowers102Dataset",
            "train_params":dict(
                root=config.input.root_path,
                classification_transform=self.classification_transforms,
                diffusion_transform=self.diffusion_transforms,
                test_classification_transform=self.test_classification_transforms,
                test_diffusion_transform=self.test_diffusion_transforms,
                subsample=config.input.subsample,
            ),
        }

        self.FGVCAircraftDataset = {   
            "target": "dataset.dataset_class_label.FGVCAircraftDataset",
            "train_params":dict(
                root=config.input.root_path,
                classification_transform=self.classification_transforms,
                diffusion_transform=self.diffusion_transforms,
                test_classification_transform=self.test_classification_transforms,
                test_diffusion_transform=self.test_diffusion_transforms,
                subsample=config.input.subsample,
            ),
        }

        self.OxfordIIITPetDataset = {   
            "target": "dataset.dataset_class_label.OxfordIIITPetDataset",
            "train_params":dict(
                root=config.input.root_path,
                classification_transform=self.classification_transforms,
                diffusion_transform=self.diffusion_transforms,
                test_classification_transform=self.test_classification_transforms,
                test_diffusion_transform=self.test_diffusion_transforms,
                subsample=config.input.subsample,
            ),
        }

        self.STL10Dataset = {   
            "target": "dataset.dataset_class_label.STL10Dataset",
            "train_params":dict(
                root=config.input.root_path,
                classification_transform=self.classification_transforms,
                diffusion_transform=self.diffusion_transforms,
                test_classification_transform=self.test_classification_transforms,
                test_diffusion_transform=self.test_diffusion_transforms,
                subsample=config.input.subsample,
            ),
        }

        self.CIFAR10Dataset = {   
            "target": "dataset.dataset_class_label.CIFAR10Dataset",
            "train_params":dict(
                root=config.input.root_path,
                classification_transform=self.classification_transforms,
                diffusion_transform=self.diffusion_transforms,
                test_classification_transform=self.test_classification_transforms,
                test_diffusion_transform=self.test_diffusion_transforms,
                subsample=config.input.subsample,
            ),
        }

        self.CIFAR100Dataset = {   
            "target": "dataset.dataset_class_label.CIFAR100Dataset",
            "train_params":dict(
                root=config.input.root_path,
                classification_transform=self.classification_transforms,
                diffusion_transform=self.diffusion_transforms,
                test_classification_transform=self.test_classification_transforms,
                test_diffusion_transform=self.test_diffusion_transforms,
                subsample=config.input.subsample,
            ),
        }

        self.ImageNetDataset = {   
            "target": "dataset.dataset_class_label.ImageNetDataset",
            "train_params":dict(
                root=config.input.root_path+'/ImageNet/val',
                classification_transform=self.classification_transforms,
                diffusion_transform=self.diffusion_transforms,
                test_classification_transform=self.test_classification_transforms,
                test_diffusion_transform=self.test_diffusion_transforms,
                subsample=config.input.subsample,
            ),
        }

        self.ImageNetCDataset = {   
            "target": "dataset.dataset_class_label.ImageNetCDataset",
            "train_params":dict(
                root=config.input.root_path+'/imagenet_c',
                classification_transform=self.classification_transforms,
                diffusion_transform=self.diffusion_transforms,
                test_classification_transform=self.test_classification_transforms,
                test_diffusion_transform=self.test_diffusion_transforms,
                subsample=config.input.subsample,
            ),
        }

        self.ImageNetRDataset = {   
            "target": "dataset.dataset_class_label.ImageNetRDataset",
            "train_params":dict(
                root=config.input.root_path+'/imagenet-r',
                classification_transform=self.classification_transforms,
                diffusion_transform=self.diffusion_transforms,
                test_classification_transform=self.test_classification_transforms,
                test_diffusion_transform=self.test_diffusion_transforms,
                subsample=config.input.subsample,
            ),
        }

        self.ImageNetStyleDataset = {   
            "target": "dataset.dataset_class_label.ImageNetStyleDataset",
            "train_params":dict(
                root=config.input.root_path+'/imagenet-styletransfer-v2/val',
                classification_transform=self.classification_transforms,
                diffusion_transform=self.diffusion_transforms,
                test_classification_transform=self.test_classification_transforms,
                test_diffusion_transform=self.test_diffusion_transforms,
                subsample=config.input.subsample,
            ),
        }

        self.ImageNetADataset = {   
            "target": "dataset.dataset_class_label.ImageNetADataset",
            "train_params":dict(
                root=config.input.root_path+'/imagenet-a',
                classification_transform=self.classification_transforms,
                diffusion_transform=self.diffusion_transforms,
                test_classification_transform=self.test_classification_transforms,
                test_diffusion_transform=self.test_diffusion_transforms,
                subsample=config.input.subsample,
            ),
        }
        
        self.ImageNetv2Dataset = {   
            "target": "dataset.dataset_class_label.ImageNetv2Dataset",
            "train_params":dict(
                root=config.input.root_path+'/imagenetv2-matched-frequency-format-val',
                classification_transform=self.classification_transforms,
                diffusion_transform=self.diffusion_transforms,
                test_classification_transform=self.test_classification_transforms,
                test_diffusion_transform=self.test_diffusion_transforms,
                subsample=config.input.subsample,
            ),
        }

        self.ObjectNetDataset = {
            "target": "dataset.dataset_class_label.ObjectNetDataset",
            "train_params":dict(
                root=config.input.root_path+'/ObjectNet/objectnet-1.0',
                classification_transform=self.classification_transforms,
                diffusion_transform=self.diffusion_transforms,
                test_classification_transform=self.test_classification_transforms,
                test_diffusion_transform=self.test_diffusion_transforms,
                use_dit=config.model.use_dit,
                subsample=config.input.subsample,
            ),
        }
