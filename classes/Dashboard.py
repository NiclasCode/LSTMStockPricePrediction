import json
from datetime import datetime
from enum import Enum
from pathlib import Path

import pandas as pd
import panel as pn
import torch
from panel import Column, Card, Row
from panel.widgets import CheckBoxGroup, MultiChoice, IntInput, TextInput, Checkbox, Tabulator, Button

from classes.models.DashboardParams import DashboardParams
from classes.models.ExperimentConfig import ExperimentConfig


class OptimizerType(Enum):
    Adam = torch.optim.Adam
    NAdam = torch.optim.NAdam
    Adagrad = torch.optim.Adagrad
    Adadelta = torch.optim.Adadelta

    @classmethod
    def names(cls):
        return [e.name for cls in [cls] for e in cls]


class Dashboard:

    def __init__(self, available_features: list, exp_config: ExperimentConfig):
        """

        """

        self.available_features = available_features
        self.exp_config = exp_config

        self.w_features = CheckBoxGroup(
            name="Training Features",
            options=self.available_features,
            value=[self.exp_config.target],
            inline=False
        )
        self.w_window_sizes = MultiChoice(name="window_size grid", options=[1, 5, 10, 20, 30, 60, 90], value=[30])
        self.w_horizon = MultiChoice(name="horizon", options=[1, 5, 10, 30, 90], value=[1])
        self.w_batch_sizes = MultiChoice(name="batch_size grid", options=[8, 16, 32, 64], value=[16])
        self.w_patiences = MultiChoice(name="patience grid", options=[5, 10, 50], value=[10])
        self.w_hidden_sizes = MultiChoice(name="hidden_size grid", options=[10, 30, 50, 100, 150, 200], value=[50])
        self.w_num_layers = MultiChoice(name="num_layers grid", options=[1, 2, 3, 4], value=[1])
        self.w_dropouts = MultiChoice(name="dropout grid", options=[0.0, 0.1, 0.2, 0.3], value=[0.0, 0.1])
        self.w_lrs = MultiChoice(name="lr grid", options=[1e-2, 1e-3, 5e-4, 1e-4], value=[1e-3, 1e-4])

        self.w_opts = MultiChoice(name="optimizer grid", options=list(OptimizerType.names()), value=["Adam"])

        # --- Repro / runtime ---
        self.w_seeds = IntInput(name="Number of seeds (replications)", value=1, start=1, end=50)
        self.w_max_epochs = IntInput(name="max_epochs", value=250, start=10, end=10000, step=50)

        # --- Artifacts ---
        self.w_runs_dir = TextInput(name="runs output dir",
                                    value=f"runs/{exp_config.target}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        self.w_save_plots = Checkbox(name="Save plots per run", value=True)
        self.w_plot_unscaled = Checkbox(name="Also save unscaled plots (inverse transform)", value=True)

        # --- Outputs ---
        self.out_status = pn.pane.Markdown("Ready.")
        self.out_table = Tabulator(pd.DataFrame(), height=300, pagination="local", page_size=20)
        self.out_best = Tabulator(pd.DataFrame(), height=250, pagination="local", page_size=10)
        self.w_results_root = TextInput(name="results root", value=str(self.exp_config.save_path))
        self.w_refresh_results = Button(name="Refresh Results", button_type="primary")

        self.dashboard = None

    def build_dashboard(self) -> pn.layout.base.ListPanel:
        """
        Builds and returns a dashboard layout for experiment configuration and result visualization.
        This dashboard includes input and output widgets organized in collapsible panels.
        The input panel allows users to configure data and grid search parameters, while the output
        panel displays execution status and results.

        :return: A Panel ListPanel layout containing input and output sections of the dashboard.
        """
        w_inputs = Row(
            Card(
                pn.pane.Markdown("## Data & Features"),
                self.w_features,
                title="Dataset",
                collapsed=False
            ),
            Card(
                pn.pane.Markdown("## Grid Search Parameters"),
                pn.Row(self.w_window_sizes, self.w_horizon),
                pn.Row(self.w_batch_sizes, self.w_patiences),
                pn.Row(self.w_hidden_sizes, self.w_num_layers),
                pn.Row(self.w_dropouts, self.w_lrs),
                pn.Row(self.w_opts),
                pn.layout.Divider(),
                pn.Row(self.w_seeds, self.w_max_epochs),
                pn.layout.Divider(),
                pn.Row(self.w_runs_dir, self.w_save_plots, self.w_plot_unscaled),
                title="Experiment Grid",
                collapsed=False
            )
        )
        self.w_refresh_results.on_click(self._on_refresh_results)
        self.refresh_results()

        w_outputs = Column(self.out_status, self.out_table, self.out_best)
        results_controls = Row(self.w_results_root, self.w_refresh_results)
        self.dashboard = Column(w_inputs, results_controls, w_outputs)
        return self.dashboard

    def serve(self):
        return self.dashboard.servable()

    def get_values(self):
        return DashboardParams(
            features=self.w_features.value,
            window_sizes=self.w_window_sizes.value,
            horizons=self.w_horizon.value,
            batch_sizes=self.w_batch_sizes.value,
            patience=self.w_patiences.value,
            hidden_sizes=self.w_hidden_sizes.value,
            num_layers=self.w_num_layers.value,
            lrs=self.w_lrs.value,
            dropouts=self.w_dropouts.value,
            optimizers=[OptimizerType[opt].value for opt in self.w_opts.value],
        )

    def _on_refresh_results(self, _event=None):
        self.refresh_results()

    def refresh_results(self) -> None:
        results_root = Path(self.w_results_root.value)
        runs_df, best_df, status = self._load_results(results_root)
        self.out_table.value = runs_df
        self.out_best.value = best_df
        self.out_status.object = status

    @staticmethod
    def _safe_read_json(path: Path) -> dict:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _load_results(self, results_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
        if not results_root.exists():
            return pd.DataFrame(), pd.DataFrame(), f"Results root not found: `{results_root}`"

        run_rows = []
        best_rows = []

        for exp_dir in sorted([p for p in results_root.iterdir() if p.is_dir()]):
            best_dir = exp_dir / "best_run"
            if best_dir.exists():
                metrics_path = best_dir / "metrics.json"
                metrics = self._safe_read_json(metrics_path) if metrics_path.exists() else {}
                best_rows.append({
                    "experiment": exp_dir.name,
                    "best_run_path": str(best_dir),
                    "has_checkpoint": (best_dir / "checkpoint.pt").exists(),
                    "has_predictions": (best_dir / "predictions.png").exists(),
                    "has_predictions_scaled": (best_dir / "predictions_scaled.png").exists(),
                    "loss": metrics.get("loss"),
                    "mae": metrics.get("mae"),
                    "rmse": metrics.get("rmse"),
                    "r2": metrics.get("r2"),
                    "directional_accuracy": metrics.get("directional_accuracy"),
                })

            for run_dir in sorted([p for p in exp_dir.iterdir() if p.is_dir() and p.name != "best_run"]):
                config_path = run_dir / "config.json"
                config = self._safe_read_json(config_path) if config_path.exists() else {}
                run_rows.append({
                    "experiment": exp_dir.name,
                    "run_name": run_dir.name,
                    "run_path": str(run_dir),
                    "window_size": config.get("window_size"),
                    "horizon": config.get("horizon"),
                    "batch_size": config.get("batch_size"),
                    "patience": config.get("patience"),
                    "hidden_size": config.get("hidden_size"),
                    "num_layers": config.get("num_layers"),
                    "dropout": config.get("dropout"),
                    "lr": config.get("lr"),
                    "optimizer": config.get("optimizer"),
                    "seed": config.get("seed"),
                    "has_checkpoint": (run_dir / "checkpoint.pt").exists(),
                    "has_loss_curve": (run_dir / "loss_curve.png").exists(),
                })

        status = f"Loaded {len(run_rows)} runs across {len(best_rows)} experiments."
        return pd.DataFrame(run_rows), pd.DataFrame(best_rows), status
