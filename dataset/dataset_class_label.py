"""Define classification datasets.

For these datasets, there are two variables that will be used
by our TTA module:
  - self.classes: ImageNet-A, ImageNet-R, and ObjectNet only include
                     a subset of ImageNet classes, e.g. N out of 1000.
                     For these datasets, `dataset.classes` will be a list
                     of length N, where each element is the selected class
                     index of the original ImageNet.
                     Otherwise, `dataset.classes` will range from 0 to number
                     of classes in the dataset.
  - self.class_names: The corresponding category names of selected classes

For subsampling, we rank file names within each class and pick the first N.

For transformations, we use `self.class_transform` and `self.diffusion_transform`
to control image transformation for the classifier and diffusion model.
"""
import json
import os
import glob
import random
import json
import pickle
from collections import defaultdict

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.datasets import VisionDataset
from datasets import load_dataset

from torchvision.datasets import (
    Food101,
    OxfordIIITPet,
    FGVCAircraft,
    CIFAR10,
    CIFAR100,
    Flowers102,
)


# Fetch meta information for each dataset
DIR = os.path.dirname(os.path.realpath(__file__))
IMAGENET_CLASS_INDEX_PATH = os.path.join(DIR, '../class_indices/imagenet_class_index.json')
OBJECTNET_CLASS_INDEX_PATH = os.path.join(DIR, '../class_indices/object_imagenet.json')
FOOD101_CLASS_INDEX_PATH = os.path.join(DIR, '../class_indices/food101_class_index.json')
OXFORDPET_CLASS_INDEX_PATH = os.path.join(DIR, '../class_indices/oxfordpet_class_index.json')
FGVCAIRCRAFT_CLASS_INDEX_PATH = os.path.join(DIR, '../class_indices/fgvc_aircraft_class_index.json')
FLOWERS_CLASS_INDEX_PATH = os.path.join(DIR, '../class_indices/flowers102_class_index.json')

with open(IMAGENET_CLASS_INDEX_PATH, 'r') as f:
    IMAGENET_CLASS_INDEX = json.load(f)

with open(OBJECTNET_CLASS_INDEX_PATH, 'r') as f:
    OBJECTNET_CLASS_INDEX = json.load(f)

with open(FOOD101_CLASS_INDEX_PATH, 'r') as f:
    FOOD101_CLASS_INDEX = json.load(f)

with open(OXFORDPET_CLASS_INDEX_PATH, 'r') as f:
    OXFORDPET_CLASS_INDEX = json.load(f)

with open(FGVCAIRCRAFT_CLASS_INDEX_PATH, 'r') as f:
    FGVCAIRCRAFT_CLASS_INDEX = json.load(f)

with open(FLOWERS_CLASS_INDEX_PATH, 'r') as f:
    FLOWERS_CLASS_INDEX = json.load(f)


################## Classification Datasets ##################
class ImageNetBase(VisionDataset):
    def __init__(
            self,
            root,
            split='5',
            classification_transform=lambda x: x,
            diffusion_transform=lambda x: x,
            test_classification_transform=lambda x: x,
            test_diffusion_transform=lambda x: x,
            subsample=None,
    ):
        super().__init__(root)
        self.class_transform = classification_transform
        self.diffusion_transform =  diffusion_transform 
        self.test_classification_transform = test_classification_transform
        self.test_diffusion_transform =  test_diffusion_transform         
        self.classes = None 
        self.class_names = []

        self.root = self._set_root(root, split)
        glob_path = self._set_glob_path()
        self._images = sorted(glob.glob(glob_path))

        self.do_objectnet = False

        # use a fixed shuflled order
        class_name = self.__class__.__name__

        # subsample dataset
        if subsample is not None:
            self._images = self._subsample(self._images, subsample)

        # Find class index mappings
        class_index_mapping = json.load(open(IMAGENET_CLASS_INDEX_PATH,'r'))
        new_class_index_mapping = {}
        index = 0
        for key,val in class_index_mapping.items():
            new_class_index_mapping[val[0]] =[key,val[1]]
            assert int(key) == index
            self.class_names.append(val[1])
            index +=1

        self.class_index_mapping = new_class_index_mapping

        self.classes = [int(new_class_index_mapping[val.split('/')[-1]][0]) for val in  glob.glob(self.root + '/n*')]
        self.classes.sort()

    def _set_root(self, root, split):
        raise NotImplementedError

    def _set_glob_path(self):
        raise NotImplementedError

    def _subsample(self, images, num):
        tabs = {}
        for i, img in enumerate(images):
            cur_class = img.split('/')[-2]
            if cur_class not in tabs:
                tabs[cur_class] = []
            tabs[cur_class].append(i)

        new_images = []
        for k, vs in tabs.items():
            for i in range(num):
                new_images.append(images[vs[i]])

        return new_images

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        filepath = self._images[index]
        class_image = self.class_transform(Image.open(filepath).convert("RGB"))
        diff_image = self.diffusion_transform(Image.open(filepath).convert("RGB"))

        test_class_image = self.test_classification_transform(Image.open(filepath).convert("RGB"))
        test_diff_image = self.test_diffusion_transform(Image.open(filepath).convert("RGB"))
        
        class_index = self.class_index_mapping[filepath.split("/")[-2]][0]
        class_index = int(class_index)
        return {"image_disc":class_image,"image_gen": diff_image, "test_image_disc":test_class_image, "test_image_gen": test_diff_image, "class_idx":class_index, 'filepath': filepath, "index": index}


class ImageNetDataset(ImageNetBase):
    def __init__(
            self,
            root,
            split='',
            test_classification_transform=lambda x: x,
            test_diffusion_transform=lambda x: x,
            classification_transform=lambda x: x,
            diffusion_transform=lambda x: x,
            subsample=None,
    ):
        super().__init__(root, split, classification_transform, diffusion_transform,
                         test_classification_transform, test_diffusion_transform,
                         subsample)

    def _set_root(self, root, split):
        return root

    def _set_glob_path(self):
        return os.path.join(self.root, '*/*.JPEG')


class ImageNetCDataset(ImageNetBase):
    def __init__(
            self,
            root,
            split='5',
            classification_transform=lambda x: x,
            diffusion_transform=lambda x: x,
            test_classification_transform=lambda x: x,
            test_diffusion_transform=lambda x: x,
            subsample=None,
    ):
        super().__init__(root=root,
                         split=split,
                         classification_transform=classification_transform,
                         diffusion_transform=diffusion_transform,
                         test_classification_transform=test_classification_transform,
                         test_diffusion_transform=test_diffusion_transform,
                         subsample=subsample,
                         )

    def _set_root(self, root, split):
        return os.path.join(root, 'gaussian_noise/' + split)

    def _set_glob_path(self):
        return os.path.join(self.root, '*/*.JPEG')


class ImageNetRDataset(ImageNetBase):
    def __init__(
            self,
            root,
            split=None,
            classification_transform=lambda x: x,
            diffusion_transform=lambda x: x,
            test_classification_transform=lambda x: x,
            test_diffusion_transform=lambda x: x,
            subsample=None,
    ):
        super().__init__(root=root,
                         split=split,
                         classification_transform=classification_transform,
                         diffusion_transform=diffusion_transform,
                         test_classification_transform=test_classification_transform,
                         test_diffusion_transform=test_diffusion_transform,
                         subsample=subsample)


    def _set_root(self, root, split):
        return root

    def _set_glob_path(self):
        return os.path.join(self.root, '*/*.jpg')


class ImageNetADataset(ImageNetBase):
    def __init__(
            self,
            root,
            split=None,
            classification_transform=lambda x: x,
            diffusion_transform=lambda x: x,
            test_classification_transform=lambda x: x,
            test_diffusion_transform=lambda x: x,
            subsample=None,
    ):
        super().__init__(root=root,
                         split=split,
                         classification_transform=classification_transform,
                         diffusion_transform=diffusion_transform,
                         test_classification_transform=test_classification_transform,
                         test_diffusion_transform=test_diffusion_transform,
                         subsample=subsample)


    def _set_root(self, root, split):
        return root

    def _set_glob_path(self):
        return os.path.join(self.root, '*/*.jpg')


class ImageNetv2Dataset(ImageNetBase):
    def __init__(
            self,
            root,
            split=None,
            classification_transform=lambda x: x,
            diffusion_transform=lambda x: x,
            test_classification_transform=lambda x: x,
            test_diffusion_transform=lambda x: x,
            subsample=None,
    ):
        super().__init__(root=root,
                         split=split,
                         classification_transform=classification_transform,
                         diffusion_transform=diffusion_transform,
                         test_classification_transform=test_classification_transform,
                         test_diffusion_transform=test_diffusion_transform,
                         subsample=subsample)

        self.classes = [int(val.split('/')[-1]) for val in  glob.glob(self.root + '/*')]
        self.classes.sort()

    def _set_root(self, root, split):
        return root

    def _set_glob_path(self):
        return os.path.join(self.root, '*/*.jpeg')

    def __getitem__(self, index):
        filepath = self._images[index]
        class_image = self.class_transform(Image.open(filepath).convert("RGB"))
        diff_image = self.diffusion_transform(Image.open(filepath).convert("RGB"))

        test_class_image = self.test_classification_transform(Image.open(filepath).convert("RGB"))
        test_diff_image = self.test_diffusion_transform(Image.open(filepath).convert("RGB"))

        
        class_index = filepath.split("/")[-2]
        class_index = int(class_index)
        return {"image_disc":class_image,"image_gen": diff_image,
                "test_image_disc":test_class_image, "test_image_gen": test_diff_image,
                "class_idx":class_index, 'filepath': filepath, "index": index}


class ImageNetStyleDataset(ImageNetBase):
    def __init__(
            self,
            root,
            split=None,
            classification_transform=lambda x: x,
            diffusion_transform=lambda x: x,
            test_classification_transform=lambda x: x,
            test_diffusion_transform=lambda x: x,
            subsample=None,
    ):
        super().__init__(root=root,
                         split=split,
                         classification_transform=classification_transform,
                         diffusion_transform=diffusion_transform,
                         test_classification_transform=test_classification_transform,
                         test_diffusion_transform=test_diffusion_transform,
                         subsample=subsample)


    def _set_root(self, root, split):
        return root

    def _set_glob_path(self):
        return os.path.join(self.root, '*/*.png')


class ObjectNetDataset(VisionDataset):
    def __init__(
            self,
            root,
            classification_transform=lambda x: x,
            diffusion_transform=lambda x: x,
            test_classification_transform=lambda x: x,
            test_diffusion_transform=lambda x: x,
            use_dit=False,
            subsample=None,
    ):
        super().__init__(root)
        self.class_transform = classification_transform
        self.diffusion_transform =  diffusion_transform 
        self.test_classification_transform = test_classification_transform
        self.test_diffusion_transform =  test_diffusion_transform         
        self.classes = None 
        self.class_names = []
        
        self.use_dit = use_dit
        
        self.imagenet_class_index_mapping = json.load(open(IMAGENET_CLASS_INDEX_PATH,'r'))

        self.root = self._set_root(root)
        glob_path = self._set_glob_path()
        
        self._images = sorted(glob.glob(glob_path))

        with open(OBJECTNET_CLASS_INDEX_PATH, 'r') as fin:
            self.class_index_mapping = json.load(fin)

        filtered_images_dict = defaultdict(lambda: [])
        images = []
        for img in self._images:
            if img.split('/')[-2] in self.class_index_mapping.keys():
                filtered_images_dict[img.split('/')[-2]].append(img)
                images.append(img)
                
        if subsample is not None:
            images = []
            for key, val in filtered_images_dict.items():
                images.extend(val[:subsample])
        self._images = images

        self.do_objectnet = False

        # use a fixed shuflled order
        class_name = self.__class__.__name__

        # subsample dataset
        if subsample is not None:
            self._images = self._subsample(self._images, subsample)

        self.class_names = list(self.class_index_mapping.keys())
        self.class_names.sort()
        
        self.classes = []
        self.all_class_names = []
        for key, vals in self.class_index_mapping.items():
            for val in vals:
                self.classes.append(int(val[0]))
                self.all_class_names.append(val[-1])
        
        self.classes.sort()
        if not self.use_dit:
            self.classes = None

    def crop_image(self, image, border=2):
        return ImageOps.crop(image, border=border)

    def _set_root(self, root):
        return root + '/images'

    def _set_glob_path(self):
        return os.path.join(self.root, './*/*.png')

    def _subsample(self, images, num):
        tabs = {}
        for i, img in enumerate(images):
            cur_class = img.split('/')[-2]
            if cur_class not in tabs:
                tabs[cur_class] = []
            tabs[cur_class].append(i)

        new_images = []
        for k, vs in tabs.items():
            for i in range(num):
                new_images.append(images[vs[i]])

        return new_images

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        filepath = self._images[index]
        
        # Remove boundary of ObjectNet images
        class_image = self.class_transform(
            self.crop_image(Image.open(filepath).convert("RGB"))
        )
        diff_image = self.diffusion_transform(
            self.crop_image(Image.open(filepath).convert("RGB"))
        )

        test_class_image = self.test_classification_transform(
            self.crop_image(Image.open(filepath).convert("RGB"))
        )
        test_diff_image = self.test_diffusion_transform(
            self.crop_image(Image.open(filepath).convert("RGB"))
        )
        
        # ObjectNet categories would overlap multiple ImageNet categories
        # return a list of integers instead
        category_name = filepath.split("/")[-2]
        if self.use_dit:
            class_index = [
                int(cat_info[0]) for cat_info in self.class_index_mapping[category_name]
            ]
        else:
            class_index = self.class_names.index(category_name)
        
        return {"image_disc":class_image,"image_gen": diff_image,
                "test_image_disc":test_class_image, "test_image_gen": test_diff_image,
                "class_idx":class_index, 'filepath': filepath, "index": index}


class Food101Dataset(Food101):
    def __init__(
            self,
            root,
            split='test',
            classification_transform=lambda x: x,
            diffusion_transform=lambda x: x,
            test_classification_transform=lambda x: x,
            test_diffusion_transform=lambda x: x,
            download=True,
            subsample=None,
    ):  
        # These dataset should already be arranged in order
        super().__init__(root=root, split=split,
                         transform=None, target_transform=None,
                         download=download)
        self.class_transform = classification_transform
        self.diffusion_transform =  diffusion_transform      
        self.test_classification_transform = test_classification_transform
        self.test_diffusion_transform = test_diffusion_transform

        class_name = self.__class__.__name__
        
        # Meta information for CLIP embeddings
        self.classes = None
        class_index_mapping = FOOD101_CLASS_INDEX
        self.class_names = ['' for _ in range(len(class_index_mapping))]
        for k in class_index_mapping.keys():
            self.class_names[int(k)] = class_index_mapping[k]
        assert not any([name == '' for name in self.class_names])

        # subsample dataset
        if subsample is not None:
            self._image_files, self._labels = self._subsample(
                self._image_files, self._labels, subsample
            )

    def _subsample(self, images, labels, num):
        tabs = {}
        for i, lab in enumerate(labels):
            if lab not in tabs:
                tabs[lab] = []
            tabs[lab].append(i)

        new_images = []
        new_labels = []
        for k, vs in tabs.items():
            for i in range(num):
                new_images.append(images[vs[i]])
                new_labels.append(labels[vs[i]])

        return new_images, new_labels

    def __getitem__(self, index):
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image_file, class_index = self._image_files[index], self._labels[index]
        image = Image.open(image_file).convert("RGB")

        class_image = self.class_transform(image)
        diff_image = self.diffusion_transform(image)

        test_class_image = self.test_classification_transform(image)
        test_diff_image = self.test_diffusion_transform(image)

        image_file = str(image_file)

        return {"image_disc":class_image,"image_gen": diff_image,
                "test_image_disc":test_class_image,"test_image_gen": test_diff_image,
                "class_idx":class_index, 'filepath':image_file, "index": index}


class OxfordIIITPetDataset(OxfordIIITPet):
    def __init__(
            self,
            root,
            split='test',
            classification_transform=lambda x: x,
            diffusion_transform=lambda x: x,
            test_classification_transform=lambda x: x,
            test_diffusion_transform=lambda x: x,
            download=True,
            subsample=None,
    ):  
        # These dataset should already be arranged in order
        super().__init__(root=root, split=split,
                         transform=None, target_transform=None,
                         download=download)
        self.class_transform = classification_transform
        self.diffusion_transform =  diffusion_transform      
        self.test_classification_transform = test_classification_transform
        self.test_diffusion_transform = test_diffusion_transform

        class_name = self.__class__.__name__
        
        # Meta information for CLIP embeddings
        self.classes = None
        class_index_mapping = OXFORDPET_CLASS_INDEX
        self.class_names = ['' for _ in range(len(class_index_mapping))]
        for k in class_index_mapping.keys():
            self.class_names[int(k)] = class_index_mapping[k]
        assert not any([name == '' for name in self.class_names])

            # subsample dataset
        if subsample is not None:
            self._images, self._labels= self._subsample(
                self._images, self._labels, subsample
            )

    def _subsample(self, images, labels, num):
        tabs = {}
        for i, lab in enumerate(labels):
            if lab not in tabs:
                tabs[lab] = []
            tabs[lab].append(i)

        new_images = []
        new_labels = []
        for k, vs in tabs.items():
            for i in range(num):
                new_images.append(images[vs[i]])
                new_labels.append(labels[vs[i]])

        return new_images, new_labels

    def __getitem__(self, index):
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        filepath = self._images[index]
        image = Image.open(self._images[index]).convert("RGB")
        class_index = self._labels[index]

        class_image = self.class_transform(image)
        diff_image = self.diffusion_transform(image)

        test_class_image = self.test_classification_transform(image)
        test_diff_image = self.test_diffusion_transform(image)

        filepath = str(filepath)

        return {"image_disc":class_image,"image_gen": diff_image,
                "test_image_disc":test_class_image,"test_image_gen": test_diff_image,
                "class_idx":class_index, 'filepath':filepath, "index": index}


class FGVCAircraftDataset(FGVCAircraft):
    def __init__(
            self,
            root,
            split='test',
            classification_transform=lambda x: x,
            diffusion_transform=lambda x: x,
            test_classification_transform=lambda x: x,
            test_diffusion_transform=lambda x: x,
            download=True,
            subsample=None,
    ):  
        # These dataset should already be arranged in order
        super().__init__(root=root, split=split,
                         transform=None, target_transform=None,
                         download=download)
        self.class_transform = classification_transform
        self.diffusion_transform =  diffusion_transform      
        self.test_classification_transform = test_classification_transform
        self.test_diffusion_transform = test_diffusion_transform

        class_name = self.__class__.__name__
        
        # Meta information for CLIP embeddings
        self.classes = None
        class_index_mapping = FGVCAIRCRAFT_CLASS_INDEX
        self.class_names = ['' for _ in range(len(class_index_mapping))]
        for k in class_index_mapping.keys():
            self.class_names[int(k)] = class_index_mapping[k]
        assert not any([name == '' for name in self.class_names])

         # subsample dataset
        if subsample is not None:
            self._image_files, self._labels = self._subsample(
                self._image_files, self._labels, subsample
            )

    def _subsample(self, images, labels, num):
        tabs = {}
        for i, lab in enumerate(labels):
            if lab not in tabs:
                tabs[lab] = []
            tabs[lab].append(i)

        new_images = []
        new_labels = []
        for k, vs in tabs.items():
            for i in range(num):
                new_images.append(images[vs[i]])
                new_labels.append(labels[vs[i]])

        return new_images, new_labels

    def __getitem__(self, index):
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image_file, class_index = self._image_files[index], self._labels[index]
        image = Image.open(image_file).convert("RGB")

        class_image = self.class_transform(image)
        diff_image = self.diffusion_transform(image)

        test_class_image = self.test_classification_transform(image)
        test_diff_image = self.test_diffusion_transform(image)

        image_file = str(image_file)

        return {"image_disc":class_image,"image_gen": diff_image,
                "test_image_disc":test_class_image,"test_image_gen": test_diff_image,
                "class_idx":class_index, 'filepath':image_file, "index": index}


class CIFAR10Dataset(CIFAR10):
    def __init__(
            self,
            root,
            train=False,
            classification_transform=lambda x: x,
            diffusion_transform=lambda x: x,
            test_classification_transform=lambda x: x,
            test_diffusion_transform=lambda x: x,
            download=True,
            subsample=None,
    ):  
        # These dataset should already be arranged in order
        super().__init__(root=root, train=train,
                         transform=None, target_transform=None,
                         download=download)
        self.class_transform = classification_transform
        self.diffusion_transform =  diffusion_transform      
        self.test_classification_transform = test_classification_transform
        self.test_diffusion_transform = test_diffusion_transform

        class_name = self.__class__.__name__
        
        self.class_names = self.classes
        self.classes = None

        # subsample dataset
        if subsample is not None:
            self.data, self.targets = self._subsample(self.data, self.targets, subsample)

    def _subsample(self, images, labels, num):
        tabs = {}
        for i, lab in enumerate(labels):
            if lab not in tabs:
                tabs[lab] = []
            tabs[lab].append(i)

        new_images = []
        new_labels = []
        for k, vs in tabs.items():
            for i in range(num):
                new_images.append(images[vs[i]])
                new_labels.append(labels[vs[i]])

        new_images = np.array(new_images)

        return new_images, new_labels

    
    def __getitem__(self, index):
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image = Image.fromarray(self.data[index])
        class_index = self.targets[index]

        class_image = self.class_transform(image)
        diff_image = self.diffusion_transform(image)

        test_class_image = self.test_classification_transform(image)
        test_diff_image = self.test_diffusion_transform(image)

        filename = f"{index:05d}"

        return {"image_disc":class_image,"image_gen": diff_image,
                "test_image_disc":test_class_image,"test_image_gen": test_diff_image,
                "class_idx":class_index, 'filepath':filename, "index": index}


class CIFAR100Dataset(CIFAR100):
    def __init__(
            self,
            root,
            train=False,
            classification_transform=lambda x: x,
            diffusion_transform=lambda x: x,
            test_classification_transform=lambda x: x,
            test_diffusion_transform=lambda x: x,
            download=True,
            subsample=None,
    ):  
        # These dataset should already be arranged in order
        super().__init__(root=root, train=train,
                         transform=None, target_transform=None,
                         download=download)
        self.class_transform = classification_transform
        self.diffusion_transform =  diffusion_transform      
        self.test_classification_transform = test_classification_transform
        self.test_diffusion_transform = test_diffusion_transform

        class_name = self.__class__.__name__
        
        self.class_names = self.classes
        self.classes = None

        # subsample dataset
        if subsample is not None:
            self.data, self.targets = self._subsample(self.data, self.targets, subsample)

    def _subsample(self, images, labels, num):
        tabs = {}
        for i, lab in enumerate(labels):
            if lab not in tabs:
                tabs[lab] = []
            tabs[lab].append(i)

        new_images = []
        new_labels = []
        for k, vs in tabs.items():
            for i in range(num):
                new_images.append(images[vs[i]])
                new_labels.append(labels[vs[i]])

        new_images = np.array(new_images)

        return new_images, new_labels

    
    def __getitem__(self, index):
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image = Image.fromarray(self.data[index])
        class_index = self.targets[index]

        class_image = self.class_transform(image)
        diff_image = self.diffusion_transform(image)

        test_class_image = self.test_classification_transform(image)
        test_diff_image = self.test_diffusion_transform(image)

        filename = f"{index:05d}"

        return {"image_disc":class_image,"image_gen": diff_image,
                "test_image_disc":test_class_image,"test_image_gen": test_diff_image,
                "class_idx":class_index, 'filepath':filename, "index": index}


class Flowers102Dataset(Flowers102):
    def __init__(
            self,
            root,
            split='test',
            classification_transform=lambda x: x,
            diffusion_transform=lambda x: x,
            test_classification_transform=lambda x: x,
            test_diffusion_transform=lambda x: x,
            download=True,
            subsample=None,
    ):  
        # These dataset should already be arranged in order
        super().__init__(root=root, split=split,
                         transform=None, target_transform=None,
                         download=download)
        self.class_transform = classification_transform
        self.diffusion_transform =  diffusion_transform      
        self.test_classification_transform = test_classification_transform
        self.test_diffusion_transform = test_diffusion_transform

        class_name = self.__class__.__name__
        
        # Meta information for CLIP embeddings
        self.classes = None
        class_index_mapping = FLOWERS_CLASS_INDEX
        self.class_names = ['' for _ in range(len(class_index_mapping))]
        for k in class_index_mapping.keys():
            self.class_names[int(k)] = class_index_mapping[k]
        assert not any([name == '' for name in self.class_names])

        # subsample dataset
        if subsample is not None:
            self._image_files, self._labels = self._subsample(
                self._image_files, self._labels, subsample
            )

    def _subsample(self, images, labels, num):
        tabs = {}
        for i, lab in enumerate(labels):
            if lab not in tabs:
                tabs[lab] = []
            tabs[lab].append(i)

        new_images = []
        new_labels = []
        for k, vs in tabs.items():
            for i in range(num):
                new_images.append(images[vs[i]])
                new_labels.append(labels[vs[i]])

        return new_images, new_labels
 
    def __getitem__(self, index):
        image_file, class_index = self._image_files[index], self._labels[index]
        image = Image.open(image_file).convert("RGB")

        class_image = self.class_transform(image)
        diff_image = self.diffusion_transform(image)

        test_class_image = self.test_classification_transform(image)
        test_diff_image = self.test_diffusion_transform(image)

        image_file = str(image_file)

        return {"image_disc":class_image,"image_gen": diff_image,
                "test_image_disc":test_class_image,"test_image_gen": test_diff_image,
                "class_idx":class_index, 'filepath':image_file, "index": index}


