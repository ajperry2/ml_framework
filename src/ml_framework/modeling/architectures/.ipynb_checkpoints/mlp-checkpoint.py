""" A Fully connected Neural network
"""

from torch.nn import Module, Sequential, Linear, Dropout
from typing import List
from .ml_module import MLModule
from ml_framework.modeling.utilities import name_to_activation_class


class MLP(MLModule, Module):
    """ A Multi Layer Perceptron

    Takes the input through `n` layers which include a linear layer,
    an activation, and optional dropout. A single linear layer is added
    to bring the final shape to one of `output size`

    """

    def __init__(
            self,
            input_size: int,
            output_size: int,
            latent_sizes: List[int],
            activation_names: List[str],
            activation_parameters: List[List[float]],
            dropout_levels: List[float],
    ):
        super(MLP, self).__init__()
        assert (
            len(latent_sizes)
            == len(activation_names)
            == len(activation_parameters)
            == len(dropout_levels)
        ), "Must have parameters for each layer"
        num_layers = len(latent_sizes)
        current_size = input_size
        self.layers = []
        for layer_i in range(num_layers):
            # Construct layer
            # Linear Layer
            self.layers.append(Linear(current_size, latent_sizes[layer_i]))
            # Activation
            activation_class = name_to_activation_class(activation_names[layer_i])
            activation = activation_class(*activation_parameters[layer_i])
            self.layers.append(activation)
            # Dropout
            has_dropout = dropout_levels[layer_i] > 0.0
            if has_dropout:
                self.layers.append(Dropout(dropout_levels[layer_i]))
            current_size = latent_sizes[layer_i]
        # One layer with no activation to get correct size out
        self.layers.append(Linear(current_size, output_size))
        self.layers = Sequential(*self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x