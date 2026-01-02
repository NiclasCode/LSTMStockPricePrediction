from dataclasses import dataclass

from torch.utils.data import DataLoader


@dataclass
class LoaderSet:
    train_loader: DataLoader
    val_loader: DataLoader
    train_val_loader: DataLoader
    test_loader: DataLoader