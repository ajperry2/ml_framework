"""
An abstract Model class.

provides utilities for aggregating torch Modules in an algebraic way
"""
from torch.nn import Module
from torch import Tensor
from typing import List


class Ensemble(Module):
    """
    Abstract Ensemble class.

    Will aggregate a collection of models
    """

    def __init__(self, sub_models: List[Module]):
        """
        Fill class data.

        Arguments:
            sub_models (List[Module]):
                The models being aggregated, all models should take
                input of the same shape and output in the same shape
        """
        assert len(sub_models) > 0, "Require a nonempty list of models"
        self.sub_models = sub_models

    def forward(self, inputs: Tensor):
        """
        Aggregate models.

        Will return the mean output of sub_models
        """
        output = self.sub_models[0](inputs)
        for model in self.sub_models[1:]:
            output += model(inputs)
        return output / len(self.sub_models)