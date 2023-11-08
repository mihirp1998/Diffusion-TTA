## How to run the code
Run `python main.py +experiment=dit` for experiments using DiT as the diffusion backbone.  Here are the arguments which you can adapt for different datasets and hyper-parameters:

* Classifier backbone: set `model.class_arch` to `convnext_tiny`/`resnet18`/`vit_b_32`/`vit_b_16`/`vit_l_14` for ImageNet-trained classifiers
* Larger DiT: use `+experiment=dit` and set `input.sd_img_res=512`
* Optimizer: set `tta.gradient_descent.optimizer` to `adam`/`sgd`
* Learning reate: set `tta.gradient_descent.base_learning_rate` to any numerical values
* Dataset: set `input.dataset_name` to `ImageNetDataset`/`ImageNetv2Dataset`/`ImageNetRDataset`/`ImageNetCDataset`/`ImageNetADataset`/`ObjectNetDataset`/`Food101Dataset`/`Flowers102Dataset`/`FGVCAircraftDataset`/`OxfordIIITPetDataset`/`CIFAR100Dataset`.
* Total batch size: set `input.batch_size` and `tta.gradient_descent.accum_iter` where the total batch size is the multiplication of these two parameters

## How to improve performance

Empirically, we found that using larger total batch size results in more stable classification improvement.  However, it takes longer time for TTA with larger batch size.  Also, we found that some backbones are better with `sgd` optimizer than `adam` optimizer.

## Commands to get started

### ConvNext-Tiny/DiT
ConvNext-Tiny is better with `adam` optimizer

* ImageNet-R
```
python main.py +experiment=dit model.class_arch=convnext_tiny input.batch_size=12 tta.gradient_descent.accum_iter=15 input.dataset_name=ImageNetRDataset tta.gradient_descent.base_learning_rate=1e-5 tta.gradient_descent.optimizer=adam
```

* ImageNet-C
```
python main.py +experiment=dit model.class_arch=convnext_tiny input.batch_size=20 tta.gradient_descent.accum_iter=9 input.dataset_name=ImageNetCDataset tta.gradient_descent.base_learning_rate=1e-5 tta.gradient_descent.optimizer=adam input.subsample=null
```

* ImageNet-A
```
python main.py +experiment=dit model.class_arch=convnext_tiny input.batch_size=15 tta.gradient_descent.accum_iter=12 input.dataset_name=ImageNetADataset tta.gradient_descent.base_learning_rate=1e-5 tta.gradient_descent.optimizer=adam input.subsample=null
```

* ImageNet-v2
```
python main.py +experiment=dit model.class_arch=convnext_tiny input.batch_size=15 tta.gradient_descent.accum_iter=12 input.dataset_name=ImageNetv2Dataset tta.gradient_descent.base_learning_rate=1e-5 tta.gradient_descent.optimizer=adam
```

* ImageNet
```
python main.py +experiment=dit model.class_arch=convnext_tiny input.batch_size=15 tta.gradient_descent.accum_iter=12 input.dataset_name=ImageNetDataset tta.gradient_descent.base_learning_rate=1e-5 tta.gradient_descent.optimizer=adam
```

* ObjectNet
```
python main.py +experiment=dit model.class_arch=convnext_tiny input.batch_size=15 tta.gradient_descent.accum_iter=12 input.dataset_name=ObjectNetDataset tta.gradient_descent.base_learning_rate=1e-5 tta.gradient_descent.optimizer=adam
```

### ResNet-18/DiT
ResNet-18 is better with `adam` optimizer

* ImageNet-R
```
python main.py +experiment=dit model.class_arch=resnet18 input.batch_size=12 tta.gradient_descent.accum_iter=15 input.dataset_name=ImageNetRDataset tta.gradient_descent.base_learning_rate=1e-5 tta.gradient_descent.optimizer=adam
```

* ImageNet-C
```
python main.py +experiment=dit model.class_arch=resnet18 input.batch_size=20 tta.gradient_descent.accum_iter=9 input.dataset_name=ImageNetCDataset tta.gradient_descent.base_learning_rate=5e-3 tta.gradient_descent.optimizer=sgd
```

* ImageNet-A
```
python main.py +experiment=dit model.class_arch=resnet18 input.batch_size=15 tta.gradient_descent.accum_iter=12 input.dataset_name=ImageNetADataset tta.gradient_descent.base_learning_rate=1e-5 tta.gradient_descent.optimizer=adam input.subsample=null
```

* ImageNet-v2
```
python main.py +experiment=dit model.class_arch=resnet18 input.batch_size=15 tta.gradient_descent.accum_iter=12 input.dataset_name=ImageNetv2Dataset tta.gradient_descent.base_learning_rate=1e-5 tta.gradient_descent.optimizer=adam
```

* ImageNet
```
python main.py +experiment=dit model.class_arch=resnet18 input.batch_size=15 tta.gradient_descent.accum_iter=12 input.dataset_name=ImageNetDataset tta.gradient_descent.base_learning_rate=1e-5 tta.gradient_descent.optimizer=adam
```

### ViT-B-32/DiT
ViT-B-32 is better with `sgd` optimizer

* ImageNet-R
```
python main.py +experiment=dit model.class_arch=vit_b_32 input.batch_size=20 tta.gradient_descent.accum_iter=9 input.dataset_name=ImageNetRDataset tta.gradient_descent.base_learning_rate=5e-3 tta.gradient_descent.optimizer=sgd
```

* ImageNet-C
```
python main.py +experiment=dit model.class_arch=vit_b_32 input.batch_size=20 tta.gradient_descent.accum_iter=9 input.dataset_name=ImageNetCDataset tta.gradient_descent.base_learning_rate=5e-3 tta.gradient_descent.optimizer=sgd
```

* ImageNet-A
```
python main.py +experiment=dit model.class_arch=vit_b_32 input.batch_size=20 tta.gradient_descent.accum_iter=9 input.dataset_name=ImageNetADataset tta.gradient_descent.base_learning_rate=5e-3 tta.gradient_descent.optimizer=sgd input.subsample=null
```

* ImageNet-v2
```
python main.py +experiment=dit model.class_arch=vit_b_32 input.batch_size=15 tta.gradient_descent.accum_iter=12 input.dataset_name=ImageNetv2Dataset tta.gradient_descent.base_learning_rate=5e-3 tta.gradient_descent.optimizer=sgd
```

* ImageNet
```
python main.py +experiment=dit model.class_arch=vit_b_32 input.batch_size=20 tta.gradient_descent.accum_iter=9 input.dataset_name=ImageNetDataset tta.gradient_descent.base_learning_rate=5e-3 tta.gradient_descent.optimizer=sgd
```

