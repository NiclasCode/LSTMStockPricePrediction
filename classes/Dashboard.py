from datetime import datetime
from enum import Enum

import pandas as pd
import panel as pn
import torch
from panel import Column, Card, Row
from panel.widgets import CheckBoxGroup, MultiChoice, IntInput, TextInput, Checkbox, Tabulator

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
        w_outputs = Row(self.out_status, self.out_table, self.out_best)
        self.dashboard = Column(w_inputs, w_outputs)
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