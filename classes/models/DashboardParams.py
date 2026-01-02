from dataclasses import dataclass
from typing import List

import torch


@dataclass(frozen=True)
class DashboardParams:
    """
    Represents immutable parameters for configuring a dashboard.

    This class is designed to store configuration parameters that describe various
    settings for a dashboard, typically used in machine learning training workflows.
    It logically categorizes these parameters, ensuring they can be provided in
    collections for flexible configuration management.

    :ivar features: A list of feature names for the dataset used in the dashboard.
    :ivar window_sizes: A list of window sizes for analysis or modelling purposes.
    :ivar horizons: A list of horizons, representing future time steps for predictions.
    :ivar batch_sizes: A list of batch sizes, defining the amount of data processed
        at one time.
    :ivar patience: A list of patience values for early stopping criteria.
    :ivar hidden_sizes: A list of sizes for hidden layers in a neural network.
    :ivar num_layers: A list of numbers of layers in a neural network architecture.
    :ivar dropouts: A list of dropout rates to regularize neural network training.
    """
    features: List[str]
    window_sizes: List[int]
    horizons: List[int]
    batch_sizes: List[int]
    patience: List[int]
    hidden_sizes: List[int]
    num_layers: List[int]
    lrs: List[float]
    dropouts: List[float]
    optimizers: List[type]
