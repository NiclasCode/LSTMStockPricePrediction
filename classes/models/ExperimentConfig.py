from dataclasses import dataclass
from torch import nn

from sklearn.preprocessing import MinMaxScaler


@dataclass
class ExperimentConfig:
    """
    Contains all non-changeable attributes of the experiment
    
    Attributes:
        file_path: Path to the input data file
        target: Name of the target variable column
        train_test_ratio: Ratio for splitting data into training and testing sets (default: 0.8)
        train_val_ratio: Ratio for splitting training data into training and validation sets (default: 0.8)
        drop_na: Whether to drop rows with missing values (default: True)
        apply_wavelet: Whether to apply wavelet transformation (default: True)
        wavelet: Type of wavelet to use for transformation (default: "haar")
        wavelet_thr_ratio: Threshold ratio for wavelet transformation (default: 0.04)
        wavelet_mode: Mode for wavelet transformation ("soft" or "hard") (default: "soft")
        scale: Whether to apply feature scaling (default: True)
        scale_mode: Scaling strategy, either "global" or "train_only" (default: "global")
        scaler_class: Class to use for feature scaling (default: MinMaxScaler)
    """
    file_path: str
    target: str
    save_path: str = "run/"
    train_test_ratio: float = 0.8
    train_val_ratio: float = 0.8

    drop_na: bool = True
    apply_wavelet: bool = True
    wavelet: str = "haar"
    wavelet_thr_ratio: float = 0.04
    wavelet_mode: str = "soft"

    scale: bool = True
    scale_mode: str = "global"  # "global" or "train_only"
    scaler_class: type = MinMaxScaler
    criterion: nn.modules.Module = nn.modules.loss.MSELoss()
