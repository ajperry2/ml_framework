""" A Fully connected Neural network
"""

from torch.nn import (
    Module,
    Sequential,
    BatchNorm2d,
    Conv2d,
    Dropout,
    Linear,
    AvgPool2d
)

from typing import List
from .ml_module import MLModule
from ml_framework.modeling.utilities import name_to_activation_class


class _ResNetBlock(Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation_name: str = "ReLu",
        is_simple: bool = False,
        stride: int = 1
    ):
        """
        """
        super(_ResNetBlock, self).__init__()
        activation_class = name_to_activation_class(activation_name)
        self.activation = activation_class()
        self.expansion = 4 if not is_simple else 1
        bn_1 = BatchNorm2d(out_dim)
        bn_2 = BatchNorm2d(out_dim)
        if not is_simple:
            conv_1 = Conv2d(
                in_dim,
                out_dim,
                kernel_size=1, bias=False
            )
            conv_2 = Conv2d(
                out_dim,
                out_dim,
                kernel_size=3, stride=stride, padding=1, bias=False)
            conv_3 = Conv2d(
                out_dim,
                out_dim*self.expansion,
                kernel_size=1, bias=False)
            bn_3 = BatchNorm2d(out_dim*self.expansion)
            self.main_path = Sequential(*[
                conv_1, bn_1, self.activation,
                conv_2, bn_2, self.activation,
                conv_3, bn_3, self.activation
            ])
        else:
            conv_1 = Conv2d(
                in_dim,
                out_dim,
                kernel_size=3, stride=stride, padding=1, bias=False
            )
            conv_2 = Conv2d(
                out_dim,
                out_dim,
                kernel_size=3, stride=1, padding=1, bias=False)
            self.main_path = Sequential(*[
                conv_1, bn_1, self.activation,
                conv_2, bn_2, self.activation
            ])
        if in_dim != self.expansion*out_dim or stride != 1:
            self.residual_path = Sequential(
                Conv2d(
                    in_dim,
                    self.expansion*out_dim,
                    kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(self.expansion*out_dim)
            )
        else:
            self.residual_path = Sequential()

    def forward(self, x):
        out = self.main_path(x)
        out += self.residual_path(x)
        out = self.activation(out)
        return out


class ResNet(MLModule, Module):
    """ A Convolutional Classifier

    Takes the input through `n` blocks which perform one or two
    convolution steps and apply a residual connection repeatedly

    """

    def __init__(
            self,
            num_classes: int = 10,
            mode: str = "152",
            activation_name: str = "ReLU",
    ):
        super(ResNet, self).__init__()
        _valid_modes = {"18", "34", "50", "101", "512"}
        assert mode in _valid_modes, f"Invalid Mode: {mode}"
        block_repeats = {
            "18": [2, 2, 2, 2],
            "34": [3, 4, 6, 3],
            "50": [3, 4, 6, 3],
            "101": [3, 4, 23, 3],
            "512": [3, 8, 36, 3]
        }[mode]
        is_simple = {
            "18": True,
            "34": True,
            "50": False,
            "101": False,
            "512": False
        }
        strides: List[int] = [1, 2, 2, 2]
        block_targets: List[int] = [64, 128, 256, 512]
        self.layers = []
        dim = 64
        for block_i in range(len(block_repeats)):
            block_repeat = block_repeats[block_i]
            block_target = block_targets[block_i]
            stride = strides[block_i]
            for i in range(block_repeat):
                block = _ResNetBlock(
                        dim,
                        block_target,
                        stride=stride,
                        is_simple=is_simple,
                        activation_name=activation_name
                    )
                self.layers.append(block)
                dim = block_target * block.expansion
                stride = 1
        self.layers = Sequential(*self.layers)
        self.input_layer = Sequential(*[
            Conv2d(
                3,
                64,
                kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(64)
        ])
        self.pool = AvgPool2d(kernel_size=4)
        self.linear_layer = Linear(512*block.expansion, num_classes)

    def forward(self, x):
        out = self.input_layer(x)
        out = self.layers(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear_layer(out)
        return out