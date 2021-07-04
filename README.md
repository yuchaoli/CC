# Towards Compact CNNs via Collaborative Compression

PyTorch implementation for Towards Compact CNNs via Collaborative Compression (CVPR2021).

## Running Code

In this code, you can run ResNet/DenseNet/VGGNet/GoogLeNet model on CIFAR10/ImageNet2012 dataset. The code has been tested by Python 3.6.8, [Pytorch 1.7.1](https://pytorch.org/) and CUDA 10.0.

## Running Example

### Train

#### CIFAR-10

##### ResNet-56 (compression ratio = 50%)

```shell
python compress.py  --dataset cifar10 \
                    --net resnet56 \
                    --pretrained True \
                    --checkpoint pth/resnet56.pth \
                    --train_dir tmp/resnet56_CC_0.5 \
                    --train_batch_size 128 \
                    --com_ratio 0.5
```
or

```shell
sh compress.sh
```

##### DenseNet-40 (compression ratio = 50%)

```shell
python compress.py  --dataset cifar10 \
                    --net densenet40 \
                    --pretrained True \
                    --checkpoint pth/densenet40.pth \
                    --train_dir tmp/densenet40_CC_0.5 \
                    --train_batch_size 128 \
                    --com_ratio 0.5
```

#### ImageNet-2012

Setting ImageNet-2012 directory in dataset/imagenet.py

##### ResNet-50 (compression ratio = 50%)

```shell
python compress.py  --dataset imagenet \
                    --net resnet50 \
                    --pretrained True \
                    --checkpoint pth/resnet50.pth \ # download from torchvision
                    --train_dir tmp/resnet50_CC_0.5 \
                    --train_batch_size 256 \
                    --com_ratio 0.5
```


