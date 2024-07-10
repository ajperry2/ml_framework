"""A class which performs evaluation on a dataset."""
from torch.nn import Module
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss as Loss
from typing import Callable, Tuple, List
from torch import Tensor
import numpy as np
from random import randint
from torcheval.metrics import Metric


class Evaluator:
    """A class which will evaluate a model as you specify."""

    def evaluate(
        self,
        model: Module,
        dataloader: DataLoader,
        loss_function: Loss,
        evaluation_metrics: dict[str, Metric],
        visualize: bool = False,
        num_samples: int = 2
    ):
        """Evaluate a model.

        The data will be iterated through once, then metrics
        will be returned and visualized if requested

        Arguments:
            model (Module):
                The model being trained
            dataloader (DataLoader):
                The data the model is being evaluated with
            loss_function (Loss):
                The loss function that will be evaluated
            evaluation_metrics (dict[str, Metric]):
                Mapping from a metric name to it's function
            visualize (bool):
                Should we plot samples?
            num_samples (int):
                Number of samples to visualize

        Returns:
            Average Loss, Metrics (Tuple[float, Dict[str, float]]):
                Mapping from a metric name to it's aggregated
                value across the dataset

        TODO: Once a robust visualization function is written add to end here
        """
        training_before = model.training
        total_loss = 0.
        model.eval()
        for i, (input_batch, target_batch) in enumerate(dataloader):
            # Single optimization step
            model.zero_grad()
            output = model(input_batch)
            loss = loss_function(output, target_batch)
            total_loss += loss.sum().item()
            for metric_name, metric_class in evaluation_metrics.items():
                metric_class.update(output, target_batch)
        average_loss = total_loss / len(dataloader.dataset)
        metric_values = {
            metric_name: metric.compute() 
            for metric_name, metric in evaluation_metrics.items()
        }
        if training_before:
            model.train()
        return average_loss, metric_values
