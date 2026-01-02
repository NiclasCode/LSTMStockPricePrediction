from dataclasses import dataclass, asdict
from typing import Tuple


@dataclass(frozen=True)
class HyperparamConfig:
    """
    Encapsulates the configuration parameters for running a machine learning model.

    This class represents all the configurable parameters required to run a model,
    both related to the data pipeline and the model architecture. It is designed
    to be immutable to ensure that the configuration does not change after
    initialization. Instances of this class can be used to manage model configurations
    and group similar configurations by generating a key excluding the seed.

    :ivar features: List of feature names to be used as input for the model.
    :ivar window_size: Size of the input time window used in the model.
    :ivar horizon: Time horizon for predictions.
    :ivar batch_size: Batch size used during training or inference.
    :ivar patience: Number of epochs to wait for improvement before early stopping.
    :ivar seed: Random seed for ensuring reproducibility.
    :ivar hidden_size: Number of hidden units in each layer of the model.
    :ivar num_layers: Number of layers in the model.
    :ivar dropout: Dropout rate used for regularization in the model.
    :ivar lr: Learning rate used by the optimizer during training.
    :ivar optimizer: Optimizer class/type used in model training.
    """
    # data
    features: list
    window_size: int
    horizon: int
    batch_size: int
    patience: int
    seed: int

    # model
    hidden_size: int
    num_layers: int
    dropout: float
    lr: float
    optimizer: type  # e.g. torch.optim.Adam

    def key(self) -> Tuple:
        """Key for grouping configs across seeds (exclude seed)."""
        d = asdict(self).copy()
        d.pop("seed", None)
        return tuple(sorted(d.items()))