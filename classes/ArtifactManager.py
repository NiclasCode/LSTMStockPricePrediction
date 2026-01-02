from dataclasses import asdict

from classes.models.ExperimentConfig import ExperimentConfig
from classes.models.HyperparamConfig import HyperparamConfig
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

class ArtifactManager:
    """
    Handles experiment artifacts on disk.

    Responsibilities:
    - Create a stable run directory based on hyperparameters.
    - Persist configs and training checkpoints.
    - Save plots to a consistent location.
    """
    def __init__(self, root_dir: str | Path):
        """
        Args:
            root_dir: Base directory where all run folders are created.
        """
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def run_dir(self, cfg: HyperparamConfig) -> Path:
        """
        Build and create the per-run directory name.

        The name is derived from key hyperparameters and includes the seed to
        avoid grouping ambiguity.
        """
        key_parts = [
            f"ws{cfg.window_size}",
            f"h{cfg.horizon}",
            f"bs{cfg.batch_size}",
            f"pat{cfg.patience}",
            f"hs{cfg.hidden_size}",
            f"nl{cfg.num_layers}",
            f"do{cfg.dropout}",
            f"lr{cfg.lr}",
            f"opt{cfg.optimizer.__name__}",
            f"seed{cfg.seed}",
        ]
        d = self.root_dir / ("__".join(key_parts))
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_json(self, path: Path, obj: dict):
        """
        Serialize a dictionary as pretty-printed JSON.

        Uses `default=str` to ensure non-JSON-native types (e.g., Path) are
        converted to strings.
        """
        path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")

    def save_checkpoint(self, run_dir: Path, model: torch.nn.Module, cfg: HyperparamConfig) -> None:
        """
        Save model state, optimizer state, and config for later resuming.

        If the model wraps an internal LSTM module, save that state dict to keep
        checkpoints compact and consistent with training code.
        """
        ckpt = {
            "config": asdict(cfg),
            # store ONLY the torch modules' state dicts
            "model_state_dict": model.LSTM.state_dict() if hasattr(model, "LSTM") else model.state_dict(),
            "optimizer_state_dict": model.optimizer.state_dict() if hasattr(model, "optimizer") else None,
        }
        torch.save(ckpt, run_dir / "checkpoint.pt")
        self.save_json(run_dir / "config.json", asdict(cfg))

    def save_figure(self, fig, run_dir: Path, name: str, dpi: int = 200):
        """
        Save a matplotlib figure to the run directory and close it.
        """
        fig.savefig(run_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
