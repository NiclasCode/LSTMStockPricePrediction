from dataclasses import asdict, dataclass
import math


@dataclass
class MetricsSummary:
    loss: float
    mae: float
    rmse: float
    r2: float
    directional_accuracy: float
    naive_rmse: float
    n_samples: int
    scale: str = "unscaled"
    loss_scale: str = "scaled"

    def to_dict(self) -> dict:
        return asdict(self)
