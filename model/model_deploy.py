# coding: utf-8

from .resnet_cifar import resnet56
from .resnet_cifar_compression import resnet56_compress
from .resnet_imagenet import resnet18, resnet50
from .resnet_imagenet_compression import resnet50_compress
from .googlenet import googlenet
from .googlenet_compression import googlenet_compress
from .vgg_cifar import vggnet, vggnet16
from .vgg_cifar_compression import vggnet_compress, vggnet16_compress
from .vgg16 import vgg16
from .vgg16_compression import vgg16_compress
from .densenet_cifar import densenet40, densenet_bc_100
from .densenet_cifar_compression import densenet40_compress, densenet_bc_100_compress
from .mobilenet_v2 import mobilenet_v2, mobilenet_v2_14
from .mobilenet_v2_compression import mobilenet_v2_compress, mobilenet_v2_14_compress