""" Utilities that are useful for allowing dynamic parametarization of models
"""


from torch import nn
from torch.nn import functional
from typing import Set

# Idea, this doesn't allow people to use new activations which are added
# to pytorch, but with a large array of functions this should not happen often
activation_function_classes: Set[str] = {
    "Threshold", "Hardtanh", "Hardswish",
    "ReLU6", "ELU", "SELU", "PReLU",
    "RReLU", "GLU", "GELU", "CELU", "LogSigmoid",
    "Hardshrink", "Tanhshrink", "Softsign",
    "Softplus", "Softmin",  "Softmax", "Softshrink",
    "gumbel_softmax", "LogSoftmax", "Hardsigmoid",
    "SiLU", "Mish", "MultiheadAttention",
    "Sigmoid", "Tanh", "ReLU", "LeakyReLU"
}
activation_functions: Set[str] = {
    "threshold", "hardtanh", "hardswish",
    "relu6", "elu", "selu", "prelu",
    "rrelu", "glu", "gelu", "celu", "logsigmoid",
    "hardshrink", "tanhshrink", "softsign",
    "softplus", "softmin",  "softmax", "softshrink",
    "gumbel_softmax", "log_softmax", "hardsigmoid",
    "silu", "mish",
    "sigmoid", "tanh", "relu", "leaky_relu"
}


def name_to_activation_class(name: str):
    assert hasattr(nn, name), \
        f"Activation {name} not found"
    if name not in activation_function_classes:
        raise ValueError(f"Is {name} an activation class?")
    return getattr(nn, name)


def name_to_activation_function(name: str):
    assert hasattr(functional, name), \
        f"Activation {name} not found"
    if name not in activation_functions:
        raise ValueError(f"Is {name} an activation function?")
    return getattr(functional, name)