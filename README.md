# trytry-imagenet

ImageNet dataset is mostly used for training CNN backbone, such as Resnet, VGG, AlexNet, and etc. The dataset consists of millions of images of 1000 different kinds of objects. By training a CNN model on such large dataset, the model will become a good feature extractor to capture the feature in an image.

After the training process is completed, we use the feature extractor layer of the model trained on imagenet as a model backbone in other deeplearning models to do further tasks. Therefore, having the control of designing your own feature extractor model, and training the model on the imagenet dataset with ease give you the ability to design more customized model architecture for computer vision tasks.

## Download Dataset
In my opinion, it is hard to download the imagenet dataset fully (155G) from the original ImageNet website. There are lots of preliminary works you have to do to get the access of the dataset. Fortunately, most of the images in the imagenet dataset are from `flicker` and other websitres. You can directly download training images there. Not surprisingly, there is a open sourced [project](https://github.com/mf1024/ImageNet-Datasets-Downloader) helpes you download the dataset.
```bash
$ git clone https://github.com/mf1024/ImageNet-Datasets-Downloader.git
$ cd ImageNet-Datasets-Downloader && mkdir download
$ python downloader.py -data_root download -number_of_classes 1000 -images_per_class 500 -multiprocessing_workers 8
```

## Distributed Training
TODO

## Pretrained Models
TODO

## Reference
- https://leimao.github.io/blog/PyTorch-Distributed-Training/
- https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
