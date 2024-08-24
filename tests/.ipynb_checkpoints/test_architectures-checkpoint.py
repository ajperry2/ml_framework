import pytest
import torch


def test_mlp():
    from ml_framework.modeling.architectures.mlp import MLP
    mlp = MLP(
            input_size=4,
            output_size=10,
            latent_sizes=[1],
            activation_names=["ReLU6"],
            activation_parameters=[[]],
            dropout_levels=[0]
    )
    y = mlp(torch.randn(1, 4))
    assert all(
        [
            shape_i == shape_j
            for shape_i, shape_j
            in zip(y.shape, [1, 10])
        ]
    ), "Invalid shape from MLP"


def test_resnet():
    from ml_framework.modeling.architectures.resnet import ResNet
    test_modes = ["18", "512"]
    for test_mode in test_modes:
        resnet = ResNet(mode=test_mode, num_classes=10) 
        y = resnet(torch.randn(1, 3, 32, 32))
        assert all(
            [
                shape_i == shape_j
                for shape_i, shape_j
                in zip(y.shape, [1, 10])
            ]
        ), "Invalid shape from MLP"