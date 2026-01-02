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
    Responsible for filesystem layout, saving checkpoints, configs, plots.
    """
    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def run_dir(self, cfg: HyperparamConfig) -> Path:
        # A stable folder name without seed grouping ambiguity:
        # group key (without seed) + seed
        # Feel free to adjust naming scheme.
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
        path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")

    def save_checkpoint(self, run_dir: Path, model, cfg):
        ckpt = {
            "config": asdict(cfg),
            # store ONLY the torch modules' state dicts
            "model_state_dict": model.LSTM.state_dict() if hasattr(model, "LSTM") else model.state_dict(),
            "optimizer_state_dict": model.optimizer.state_dict() if hasattr(model, "optimizer") else None,
        }
        torch.save(ckpt, run_dir / "checkpoint.pt")
        self.save_json(run_dir / "config.json", asdict(cfg))

    def save_figure(self, fig, run_dir: Path, name: str, dpi: int = 200):
        fig.savefig(run_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
