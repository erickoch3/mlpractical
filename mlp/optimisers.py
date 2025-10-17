# -*- coding: utf-8 -*-
"""Model optimisers.

This module contains objects implementing (batched) stochastic gradient descent
based optimisation of models.
"""

import logging
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import tqdm
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class Optimiser(object):
    """Basic model optimiser."""

    def __init__(
        self,
        model: Any,
        error: Any,
        learning_rule: Any,
        train_dataset: Any,
        valid_dataset: Optional[Any] = None,
        data_monitors: Optional[Dict[str, Callable]] = None,
        notebook: bool = False,
    ) -> None:
        """Create a new optimiser instance.

        Args:
            model: The model to optimise.
            error: The scalar error function to minimise.
            learning_rule: Gradient based learning rule to use to minimise
                error.
            train_dataset: Data provider for training set data batches.
            valid_dataset: Data provider for validation set data batches.
            data_monitors: Dictionary of functions evaluated on targets and
                model outputs (averaged across both full training and
                validation data sets) to monitor during training in addition
                to the error. Keys should correspond to a string label for
                the statistic being evaluated.
        """
        self.model = model
        self.error = error
        self.learning_rule = learning_rule
        self.learning_rule.initialise(self.model.params)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.data_monitors: OrderedDict = OrderedDict([("error", error)])
        if data_monitors is not None:
            self.data_monitors.update(data_monitors)
        self.notebook = notebook
        if notebook:
            self.tqdm_progress = tqdm.tqdm_notebook
        else:
            self.tqdm_progress = tqdm.tqdm

        self.all_grads: List[List[float]] = []
        self.layers: List[List[str]] = []

    def do_training_epoch(self) -> None:
        """Do a single training epoch.

        This iterates through all batches in training dataset, for each
        calculating the gradient of the estimated error given the batch with
        respect to all the model parameters and then updates the model
        parameters according to the learning rule.
        """
        layers: List[str] = []
        all_grads: List[float] = []
        obtained_grads = False
        with self.tqdm_progress(
            total=self.train_dataset.num_batches
        ) as train_progress_bar:
            train_progress_bar.set_description("Ep Prog")
            for inputs_batch, targets_batch in self.train_dataset:
                activations = self.model.fprop(inputs_batch)
                grads_wrt_outputs = self.error.grad(activations[-1], targets_batch)
                grads_wrt_params = self.model.grads_wrt_params(
                    activations, grads_wrt_outputs
                )
                self.learning_rule.update_params(grads_wrt_params)
                train_progress_bar.update(1)
                if not obtained_grads:
                    all_grads = []
                    all_grads.extend(grads_wrt_params)
                    for i, grad in enumerate(all_grads):
                        all_grads[i] = np.abs(grad).mean()
                    layers.extend(
                        [f"Layer_{i+1}" for i in range(len(grads_wrt_params))]
                    )
                    obtained_grads = True
        self.layers.append(layers)
        self.all_grads.append(all_grads)

    def plot_grad_flow(self) -> Tuple[Figure, Axes]:
        """Plots the gradient across epochs.

        Returns:
            Tuple[Figure, Axes]: a plot of the gradient flow across epochs.
        """
        all_grads = self.all_grads
        layers = (
            self.layers[0] if self.layers else []
        )  # Take first epoch's layers as they don't change

        # Plot gradient flow for each epoch
        fig, ax = plt.subplots(figsize=(10, 6))

        for epoch_idx, epoch_grads in enumerate(all_grads):
            ax.plot(epoch_grads, alpha=0.3, color="b")

        ax.hlines(0, 0, len(layers), linewidth=1, color="k")
        ax.set_xticks(range(0, len(layers)))
        ax.set_xticklabels(layers, rotation="vertical")
        ax.set_xlim(xmin=0, xmax=len(layers))
        ax.set_xlabel("Layers")
        ax.set_ylabel("Average Gradient")
        ax.set_title("Gradient flow")
        ax.grid(True)
        plt.tight_layout()

        return fig, ax

    def eval_monitors(self, dataset: Any, label: str) -> OrderedDict:
        """Evaluates the monitors for the given dataset.

        Args:
            dataset: Dataset to perform evaluation with.
            label: Tag to add to end of monitor keys to identify dataset.

        Returns:
            OrderedDict of monitor values evaluated on dataset.
        """
        data_mon_vals = OrderedDict(
            [(key + label, 0.0) for key in self.data_monitors.keys()]
        )
        for inputs_batch, targets_batch in dataset:
            activations = self.model.fprop(inputs_batch, evaluation=True)
            for key, data_monitor in self.data_monitors.items():
                data_mon_vals[key + label] += data_monitor(
                    activations[-1], targets_batch
                )
        for key, data_monitor in self.data_monitors.items():
            data_mon_vals[key + label] /= dataset.num_batches
        return data_mon_vals

    def get_epoch_stats(self) -> OrderedDict:
        """Computes training statistics for an epoch.

        Returns:
            An OrderedDict with keys corresponding to the statistic labels and
            values corresponding to the value of the statistic.
        """
        epoch_stats = OrderedDict()
        epoch_stats.update(self.eval_monitors(self.train_dataset, "(train)"))
        if self.valid_dataset is not None:
            epoch_stats.update(self.eval_monitors(self.valid_dataset, "(valid)"))
        return epoch_stats

    def log_stats(self, epoch: int, epoch_time: float, stats: OrderedDict) -> None:
        """Outputs stats for a training epoch to a logger.

        Args:
            epoch (int): Epoch counter.
            epoch_time: Time taken in seconds for the epoch to complete.
            stats: Monitored stats for the epoch.
        """
        stats_str = ", ".join([f"{k}={round(v,2)}" for (k, v) in stats.items()])
        logger.info(
            "Epoch %d: %.2fs to complete\n    %s",
            epoch,
            epoch_time,
            stats_str,
        )

    def train(
        self, num_epochs: int, stats_interval: int = 5
    ) -> Tuple[NDArray[np.floating], Dict[str, int], float]:
        """Trains a model for a set number of epochs.

        Args:
            num_epochs: Number of epochs (complete passes through training
                dataset) to train for.
            stats_interval: Training statistics will be recorded and logged
                every `stats_interval` epochs.

        Returns:
            Tuple with first value being an array of training run statistics
            and the second being a dict mapping the labels for the statistics
            recorded to their column index in the array.
        """
        start_train_time = time.time()
        stats = self.get_epoch_stats()
        run_stats = [list(stats.values())]
        with self.tqdm_progress(total=num_epochs) as progress_bar:
            progress_bar.set_description("Exp Prog")
            for epoch in range(1, num_epochs + 1):
                start_time = time.time()
                self.do_training_epoch()
                epoch_time = time.time() - start_time
                if epoch % stats_interval == 0:
                    stats = self.get_epoch_stats()
                    self.log_stats(epoch, epoch_time, stats)
                    run_stats.append(list(stats.values()))
                progress_bar.update(1)
        finish_train_time = time.time()
        total_train_time = finish_train_time - start_train_time
        return (
            np.array(run_stats),
            {k: i for i, k in enumerate(stats.keys())},
            total_train_time,
        )
