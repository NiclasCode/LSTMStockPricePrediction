from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from classes.ArtifactManager import ArtifactManager
from classes.DataManager import DataManager
from classes.LSTMRegressor import LSTMRegressor
from classes.LiveLossPlot import LiveLossPlot
from classes.PlotFactory import PlotFactory
from classes.models.DashboardParams import DashboardParams
from classes.models.ExperimentConfig import ExperimentConfig
from classes.models.HyperparamConfig import HyperparamConfig
from classes.models.LoaderSet import LoaderSet


class ExperimentRunner:

    def __init__(self, exp_config: ExperimentConfig, device: torch.device):
        self.exp_config = exp_config
        self.hyperparam_configs = []
        self.device = device

        self.data_manager = None
        self.artifact_manager = ArtifactManager(exp_config.save_path)

        self.best_val_loss = float("inf")
        self.best_model = None
        self.best_state = None
        self.best_hyperparam_config = None
        self.loader_set = None
        self.train_losses = []
        self.val_losses = []

    def prepare(self, dashboard_params: DashboardParams):

        self.hyperparam_configs = self.build_hyperparam_configs(dashboard_params)
        print(f"Running {len(self.hyperparam_configs)} experiments")
        print(f"First 10 configs: {self.hyperparam_configs[:10]}")

        self.data_manager = DataManager(self.exp_config)
        self.data_manager.prepare(dashboard_params.features)

    def run_grid_search(self):
        for hyperparam_config in self.hyperparam_configs:
            self.run_experiment(hyperparam_config)

    def run_experiment(self, hyperparam_config: HyperparamConfig):
        print(f"Running experiment: {hyperparam_config}")

        if not hasattr(self, "live_plot"):
            self.live_plot = LiveLossPlot(update_every=5)

        self.live_plot.reset(title=str(hyperparam_config))

        self.loader_set : LoaderSet = self.data_manager.build_loaders(hyperparam_config)

        model = LSTMRegressor(
            n_features=len(hyperparam_config.features),
            hidden_size=hyperparam_config.hidden_size,
            num_layers=hyperparam_config.num_layers,
            dropout=hyperparam_config.dropout
        )

        # optional TODO: add parameter weight decay
        optimizer = hyperparam_config.optimizer(
            model.parameters(),
            lr=hyperparam_config.lr
        )

        self.train_model(model, optimizer, hyperparam_config)

    def train_model(self, model: LSTMRegressor, optimizer: torch.optim.Optimizer, hyperparam_config: HyperparamConfig, max_epochs: int = 250, patience: int = 25):

        best_val = float("inf")
        best_state = None
        wait = 0

        train_losses = []
        val_losses = []

        for epoch in range(1, max_epochs + 1):
            train_loss = self.train_one_epoch(model, self.loader_set.train_loader, optimizer)
            val_loss = self.eval_loss(model, self.loader_set.val_loader)

            # print(f"Epoch {epoch}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

            if self.live_plot is not None:
                self.live_plot.update(epoch, train_loss, val_loss)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = model.state_dict()
                wait = 0
            else:
                wait += 1

            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                print(f"Best validation loss: {best_val:.4f} at epoch {epoch - wait}")
                break

        model.load_state_dict(best_state)

        if best_val < self.best_val_loss:
            self.best_val_loss = best_val
            self.best_model = model
            self.best_state = best_state
            self.best_hyperparam_config = hyperparam_config
            self.train_losses = train_losses
            self.val_losses = val_losses

        run_dir = self.artifact_manager.run_dir(hyperparam_config)
        self.artifact_manager.save_figure(self.live_plot.fig, run_dir,
                                          "loss_curve", dpi=300)
        self.artifact_manager.save_checkpoint(run_dir, model, hyperparam_config)

        pass

    def train_one_epoch(self, model: LSTMRegressor, train_loader: DataLoader, optimizer: torch.optim.Optimizer):
        model.train()
        total, n = 0.0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            optimizer.zero_grad()

            pred = model(xb)
            loss = self.exp_config.criterion(pred, yb)

            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            total += loss.item() * bs
            n += bs

        return total / max(n, 1)

    def eval_loss(self, model: LSTMRegressor, loader: DataLoader):
        model.eval()
        total, n = 0.0, 0

        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = model(xb).to(device=self.device)
                loss = self.exp_config.criterion(pred, yb)

                bs = xb.size(0)
                total += loss.item() * bs
                n += bs

        return total / max(n, 1)

    def evaluate_best_model(self):
        loss, predictions, trues = self.predict(self.loader_set.test_loader)
        print(f"Test loss: {loss:.4f}")
        fig = PlotFactory.plot_predictions(predictions, trues)
        self.artifact_manager.save_figure(fig, self.artifact_manager.run_dir(self.best_hyperparam_config), "predictions")
        # fig = PlotFactory.plot_accuracy()

    def predict(self, loader: DataLoader):
        self.best_model.eval()

        total, n = 0.0, 0
        predictions = []
        trues = []

        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = self.best_model(xb).to(device=self.device)
                predictions.append(pred.cpu().numpy())
                trues.append(yb.cpu().numpy())

                loss = self.exp_config.criterion(pred, yb)
                bs = xb.size(0)
                total += loss.item() * bs
                n += bs

        return total / max(1, n), np.concatenate(predictions), np.concatenate(trues)



    def build_hyperparam_configs(self, dashboard_params: DashboardParams):
        print("Dashboard features:" + str(dashboard_params.features))
        features = dashboard_params.features
        if self.exp_config.target not in features:
            features.insert(0, self.exp_config.target)

        return [HyperparamConfig(
            features=dashboard_params.features,
            window_size=window_size,
            horizon=horizon,
            batch_size=batch_size,
            patience=patience,
            seed=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            lr=lr,
            optimizer=optimizer,
        )
            for (
                window_size,
                horizon,
                batch_size,
                patience,
                hidden_size,
                num_layers,
                dropout,
                lr,
                optimizer,
            ) in product(
                dashboard_params.window_sizes,
                dashboard_params.horizons,
                dashboard_params.batch_sizes,
                dashboard_params.patience,
                dashboard_params.hidden_sizes,
                dashboard_params.num_layers,
                dashboard_params.dropouts,
                dashboard_params.lrs,
                dashboard_params.optimizers,
            )]
