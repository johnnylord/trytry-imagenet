# trytry-imagenet

ImageNet dataset is mostly used for training CNN backbone, such as Resnet, VGG, AlexNet, and etc. The dataset consists of millions of images of 1000 different kinds of objects. By training a CNN model on such large dataset, the model will become a good feature extractor to capture the feature in an image.

After the training process is completed, we use the feature extractor layer of the model trained on imagenet as a model backbone in other deeplearning models to do further tasks. Therefore, having the control of designing your own feature extractor model, and training the model on the imagenet dataset with ease give you the ability to design more customized model architecture for computer vision tasks.

## Download Dataset
In my opinion, it is hard to download the imagenet dataset fully (155G) from the original ImageNet website. There are lots of preliminary works you have to do to get the access of the dataset. Fortunately, most of the images in the imagenet dataset are from `flicker` and other websitres. You can directly download training images there. Not surprisingly, there is a open [challenge](https://github.com/mf1024/ImageNet-Datasets-Downloader) on Kaggle already packages ImageNet dataset for researcher to use.
```bash
$ cd download
$ kaggle competitions download -c imagenet-object-localization-challenge
$ unzip imagenet-object-localization-challenge.zip
$ rm -rf imagenet-object-localization-challenge.zip
$ tar -xzvf imagenet_object_localization_patched2019.tar.gz
$ rm -rf imagenet_object_localization_patched2019.tar.gz
```

The directory hierarchy of dataset is as following:
```bash
ILSVRC
├── Annotations
│   └── CLS-LOC
│       ├── train   # POSCAL XML format
│       └── val     # POSCAL XML format
├── Data
│   └── CLS-LOC
│       ├── test    # testing images
│       ├── train   # training images are categorized by its parent directory
│       └── val     # validation images
└── ImageSets
    └── CLS-LOC
```

## Training
- Normal training (with one GPU)
```bash
$ python main.py --config config/resnet18.yml
```

- Distributed training (with multiple gpus)
```bash
$ python dist_main.py --config config/resnet18_dist.yml
```

## Pretrained Models
TODO

## Reference
- https://leimao.github.io/blog/PyTorch-Distributed-Training/
- https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
