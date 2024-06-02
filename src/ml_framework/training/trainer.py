"""A class which performs utilities related to training."""
from torch.nn import Module
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss as Loss
from typing import Callable, Tuple, List
from torch import Tensor
import numpy as np
from random import randint

class Trainer:
    """A class which will train a model as you specify."""

    def train(
        self,
        model: Module,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        optimizer: Optimizer,
        loss_function: Loss,
        epochs: int,
        steps_per_test: int,
        learning_rate_scheduler: lr_scheduler,
        train_callback: Callable = lambda loss: None,
        test_callback: Callable = lambda loss, samples: None,
        visualize: bool = False,
        num_samples: int = 2
    ):
        """Train a model.

        Each `steps_per_test` iterations we will log
        the average train and test loss with th train_callback
        and the test_callback function

        Arguments:
            model (Module):
                The model being trained
            train_dataloader (DataLoader):
                The data the model is being trained on
            test_dataloader (DataLoader):
                The data the model is being tested on
            optimizer (Optimizer):
                The optimizer already loaded with the
                models parameters
            loss_function (Loss):
                The loss class whose represented function
                is being optimized
            epochs (int):
                How many iterations through the data should
                be performed
            steps_per_log (int):
                How often should we perform logging?
            logging_callback (Callable):
                What should we do for logging?
            visualize (bool):
                Should we plot samples?
            num_samples (int):
                Number of samples to visualize
        """
        iterations = 0
        average_train_loss = 0.0
        model.train()

        for epoch in range(epochs):
            for i, (input_batch, target_batch) in enumerate(train_dataloader):
                # Single optimization step
                model.zero_grad()
                output = model(input_batch)
                loss = loss_function(output, target_batch)
                average_train_loss += loss.item() / input_batch.shape[0]
                loss.backward()
                optimizer.step()
                learning_rate_scheduler.step(epoch + i/len(train_dataloader))
                iterations += 1
                if iterations % steps_per_test == 0:
                    # Test
                    train_callback(average_train_loss)
                    average_loss, samples = self._test(
                        model, test_dataloader, loss_function, num_samples
                    )
                    test_callback(average_loss, samples)
        model.eval()

    def _test(
        self,
        model: Module,
        test_dataloader: DataLoader,
        loss_function: Loss,
        num_examples: int
    ) -> Tuple[float, List[Tuple[Tensor, Tensor, Tensor]]]:
        """
        Run a single evaluation of the model.

        Collect a few samples of inputs outputs so we can
        allow loggers to visualize if they want.

        Returns:
            Average Loss and examples (tuple):
                The average loss per example and list of input,
                output, target examples (all tensors)
        """
        model.eval()
        total_loss = 0
        sample_steps = np.round(
            np.linspace(0, len(test_dataloader) - 1, num_examples)
        ).astype(int)
        sample_steps = set(sample_steps.tolist())
        samples = []
        for i, (input_batch, target_batch) in enumerate(test_dataloader):
            output = model(input_batch)
            loss = loss_function(output, target_batch)
            total_loss += loss.item()
            if i in sample_steps:
                rand_sample_i = randint(0, len(input_batch)-1)
                samples.append((
                    input_batch[rand_sample_i],
                    output[rand_sample_i],
                    target_batch[rand_sample_i],
                ))
        batch_size = len(input_batch)
        average_loss = total_loss / (batch_size*len(test_dataloader))

        return average_loss, samples
