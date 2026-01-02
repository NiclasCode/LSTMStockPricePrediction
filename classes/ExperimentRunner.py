from itertools import product
from typing import List, Optional, Tuple

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
from classes.models.ModelMetrics import ModelMetrics


class ExperimentRunner:
    """
    Runs hyperparameter sweeps, training, and evaluation for LSTM experiments.

    Handles training loops, early stopping, artifact persistence, and metric
    reporting for the best-performing model.
    """

    def __init__(self, exp_config: ExperimentConfig, device: torch.device) -> None:
        """
        Initialize the experiment runner.

        Args:
            exp_config: Experiment configuration (data, preprocessing, loss).
            device: Torch device for training and evaluation.
        """
        self.exp_config = exp_config
        self.hyperparam_configs: List[HyperparamConfig] = []
        self.device = device

        self.data_manager: Optional[DataManager] = None
        self.artifact_manager = ArtifactManager(exp_config.save_path)

        self.best_val_loss = float("inf")
        self.best_model: Optional[LSTMRegressor] = None
        self.best_state: Optional[dict] = None
        self.best_hyperparam_config: Optional[HyperparamConfig] = None
        self.loader_set: Optional[LoaderSet] = None
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def prepare(self, dashboard_params: DashboardParams) -> None:
        """
        Initialize data and build the hyperparameter grid.

        Args:
            dashboard_params: Dashboard selections for features and hyperparameters.
        """

        self.hyperparam_configs = self.build_hyperparam_configs(dashboard_params)
        print(f"Running {len(self.hyperparam_configs)} experiments")
        print(f"First 10 configs: {self.hyperparam_configs[:10]}")

        self.data_manager = DataManager(self.exp_config)
        self.data_manager.prepare(dashboard_params.features)

    def run_grid_search(self) -> None:
        """
        Execute all experiments in the hyperparameter grid.
        """
        for hyperparam_config in self.hyperparam_configs:
            self.run_experiment(hyperparam_config)

    def run_experiment(self, hyperparam_config: HyperparamConfig) -> None:
        """
        Train a model for a single hyperparameter configuration.

        Args:
            hyperparam_config: Hyperparameter configuration for this run.
        """
        print(f"Running experiment: {hyperparam_config}")

        if not hasattr(self, "live_plot"):
            self.live_plot = LiveLossPlot(update_every=5)

        self.live_plot.reset(title=str(hyperparam_config))

        self.loader_set: LoaderSet = self.data_manager.build_loaders(hyperparam_config)

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

    def train_model(
        self,
        model: LSTMRegressor,
        optimizer: torch.optim.Optimizer,
        hyperparam_config: HyperparamConfig,
        max_epochs: int = 250,
        patience: int = 25,
    ) -> None:
        """
        Train one model with early stopping and record the best checkpoint.

        Args:
            model: The LSTM model to train.
            optimizer: Optimizer instance for training.
            hyperparam_config: Hyperparameters used for this run.
            max_epochs: Maximum number of epochs to train.
            patience: Early-stopping patience (epochs without improvement).
        """

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
                print(f"Best validation loss: {best_val:.6f} at epoch {epoch - wait}")
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

    def train_one_epoch(
        self,
        model: LSTMRegressor,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """
        Run a single epoch and return the average training loss.

        Args:
            model: The model to train.
            train_loader: DataLoader for the training split.
            optimizer: Optimizer instance for training.

        Returns:
            Average training loss for the epoch.
        """
        model.train()
        total, n = 0.0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            optimizer.zero_grad()

            pred = model(xb).to(self.device)
            loss = self.exp_config.criterion(pred, yb)

            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            total += loss.item() * bs
            n += bs

        return total / max(n, 1)

    def eval_loss(self, model: LSTMRegressor, loader: DataLoader) -> float:
        """
        Evaluate model loss over a loader without gradient updates.

        Args:
            model: The model to evaluate.
            loader: DataLoader for the evaluation split.

        Returns:
            Average loss over the loader.
        """
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

    def evaluate_best_model(self) -> None:
        """
        Evaluate the best model on the test set and save metrics/plots.
        """
        if self.best_hyperparam_config is None:
            raise ValueError("No best model available. Run experiments before evaluation.")
        loader_set = self.data_manager.build_loaders(self.best_hyperparam_config)
        loss, predictions, trues = self.predict(loader_set.test_loader)
        y_pred = self._inverse_scale_target(predictions)
        y_true = self._inverse_scale_target(trues)
        metrics = self.compute_metrics(y_true, y_pred, loss)
        print(metrics.to_dict())
        fig = PlotFactory.plot_predictions(y_true, y_pred)
        run_dir = self.artifact_manager.run_dir(self.best_hyperparam_config)
        self.artifact_manager.save_figure(fig, run_dir, "predictions")
        self.artifact_manager.save_json(run_dir / "metrics.json", metrics.to_dict())
        # fig = PlotFactory.plot_accuracy()

    def predict(self, loader: DataLoader) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Generate predictions and loss for a loader.

        Args:
            loader: DataLoader to generate predictions for.

        Returns:
            Tuple of (average loss, predictions array, ground-truth array).
        """
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

    def _inverse_scale_target(self, values: np.ndarray) -> np.ndarray:
        """
        Inverse-scale target values using the configured scaler.

        Args:
            values: Scaled target values.

        Returns:
            Unscaled target values.
        """
        values = np.asarray(values).reshape(-1)
        if not self.exp_config.scale or self.data_manager is None or self.data_manager.scaler is None:
            return values
        if self.data_manager.df is not None and self.data_manager.df.columns is not None:
            n_features = len(self.data_manager.df.columns)
        else:
            n_features = self.data_manager.scaled_data.shape[1]
        dummy = np.zeros((len(values), n_features), dtype=float)
        dummy[:, 0] = values
        unscaled = self.data_manager.scaler.inverse_transform(dummy)
        return unscaled[:, 0]

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, loss: float) -> ModelMetrics:
        """
        Compute standard regression metrics and directional accuracy.

        Args:
            y_true: Ground-truth values (unscaled).
            y_pred: Predicted values (unscaled).
            loss: Average loss computed on scaled values.

        Returns:
            ModelMetrics dataclass with computed metrics.
        """
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)

        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
        if denom > 0.0:
            r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)
        else:
            r2 = float("nan")

        if len(y_true) >= 2:
            actual_dir = np.sign(y_true[1:] - y_true[:-1])
            pred_dir = np.sign(y_pred[1:] - y_true[:-1])
            directional_accuracy = float(np.mean(actual_dir == pred_dir))
        else:
            directional_accuracy = float("nan")

        return ModelMetrics(
            loss=float(loss),
            mae=mae,
            rmse=rmse,
            r2=r2,
            directional_accuracy=directional_accuracy,
            n_samples=int(len(y_true)),
            scale="unscaled",
            loss_scale="scaled",
        )


    def build_hyperparam_configs(self, dashboard_params: DashboardParams) -> List[HyperparamConfig]:
        """
        Expand dashboard params into a full hyperparameter grid.

        Args:
            dashboard_params: Dashboard selections for features and hyperparameters.

        Returns:
            List of HyperparamConfig objects for the grid search.
        """
        print("Dashboard features:" + str(dashboard_params.features))
        features = list(dashboard_params.features)
        if self.exp_config.target not in features:
            features.insert(0, self.exp_config.target)

        return [HyperparamConfig(
            features=list(features),
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
