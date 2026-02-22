import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PlotFactory:
    
    @staticmethod
    def plot_loss(history: dict) -> plt.Figure:
        """Creates and displays a plot showing training and validation loss over epochs.

        Args:
            history: Dictionary containing training history with 'train_loss' and 'val_loss' lists

        Returns:
            matplotlib.figure.Figure: The generated plot figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(history['train_loss'], label='Training Loss')
        ax.plot(history['val_loss'], label='Validation Loss')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Model Loss Over Time')
        ax.legend()

        plt.show()
        return fig

    @staticmethod
    def plot_predictions(y_true, y_pred) -> plt.Figure:
        """Creates and displays a line plot comparing true vs predicted values over time.

        Args:
            y_true: Array of true/actual values
            y_pred: Array of predicted values

        Returns:
            matplotlib.figure.Figure: The generated line plot figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(y_true, label='True Values')
        ax.plot(y_pred, label='Predictions')

        ax.set_xlabel('Time')
        ax.set_ylabel('Values')
        ax.set_title('True vs Predicted Values Over Time')
        ax.legend()

        plt.show()
        return fig

    @staticmethod
    def plot_data(df: pd.DataFrame) -> None:
        """Creates and displays a line plot of the given dataframe."""
        fig, ax = plt.subplots(figsize=(12, 6))
        x = pd.to_datetime(df.index, errors="coerce")
        n = len(df)
        first_end = int(n * 0.64)
        second_end = first_end + int(n * 0.16)
        print(first_end, second_end)
        print(df.index[first_end], df.index[second_end])

        ax.plot(x[:first_end], df[:first_end], color="green")
        ax.plot(x[first_end:second_end], df[first_end:second_end], color="red")
        ax.plot(x[second_end:], df[second_end:], color="black")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        fig.autofmt_xdate()
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title('NASDAQ-100: 2005 - 2025')
        plt.show()

    def plot_equity_comparison(sim_df, starting_capital: float = None) -> plt.Figure:
        """
        Plot strategy equity vs buy-and-hold and a naive flat baseline.

        Args:
            sim_df: DataFrame returned by simulate_trading.
            starting_capital: Optional starting capital override.

        Returns:
            matplotlib.figure.Figure: The generated comparison plot.
        """
        if "equity" not in sim_df.columns or "price" not in sim_df.columns:
            raise ValueError("sim_df must contain 'equity' and 'price' columns.")

        if starting_capital is None:
            starting_capital = float(sim_df["equity"].iloc[0]) if len(sim_df) else 0.0

        prices = sim_df["price"].to_numpy(dtype=float)
        equity = sim_df["equity"].to_numpy(dtype=float)

        if len(prices) > 0 and prices[0] != 0.0:
            buy_hold = starting_capital * (prices / prices[0])
        else:
            buy_hold = np.full_like(prices, starting_capital, dtype=float)

        naive = np.full_like(prices, starting_capital, dtype=float)

        x = sim_df["date"] if "date" in sim_df.columns else np.arange(len(sim_df))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, equity, label="Strategy")
        ax.plot(x, buy_hold, label="Buy & Hold")
        ax.plot(x, naive, label="Starting capital")

        ax.set_xlabel("Date")
        ax.set_ylabel("Equity")
        ax.set_title("Equity Curve Comparison")
        ax.legend()

        plt.show()
        return fig

    @staticmethod
    def plot_shap_feature_importance(summary_df: pd.DataFrame, top_n: int = 30) -> plt.Figure:
        """
        Plot mean absolute SHAP values per feature as a horizontal bar chart.
        """
        if summary_df.empty:
            raise ValueError("summary_df is empty.")

        df = summary_df.copy()
        if top_n is not None and len(df) > top_n:
            df = df.head(top_n)

        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(df))))
        ax.barh(df["feature"][::-1], df["mean_abs_shap"][::-1])
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Feature Importance (Mean |SHAP|)")
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_shap_timestep_importance(timestep_importance: np.ndarray) -> plt.Figure:
        """
        Plot mean absolute SHAP values across timesteps.
        """
        if timestep_importance.size == 0:
            raise ValueError("timestep_importance is empty.")

        x = np.arange(1, len(timestep_importance) + 1)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x, timestep_importance, marker="o", linewidth=1.5)
        ax.set_xlabel("Timestep (1 = oldest in window)")
        ax.set_ylabel("Mean |SHAP value|")
        ax.set_title("Timestep Importance")
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_shap_feature_timestep_heatmap(
        feature_names: list,
        timestep_feature_importance: np.ndarray,
        top_n: int = 20,
    ) -> plt.Figure:
        """
        Plot a heatmap of mean absolute SHAP values by feature and timestep.
        """
        if timestep_feature_importance.size == 0:
            raise ValueError("timestep_feature_importance is empty.")

        if timestep_feature_importance.ndim != 2:
            raise ValueError("timestep_feature_importance must be 2D (timesteps x features).")

        timesteps, n_features = timestep_feature_importance.shape
        if n_features != len(feature_names):
            raise ValueError("feature_names length does not match importance matrix.")

        mean_importance = timestep_feature_importance.mean(axis=0)
        order = np.argsort(mean_importance)[::-1]
        if top_n is not None:
            order = order[:min(top_n, len(order))]

        data = timestep_feature_importance[:, order].T
        names = [feature_names[i] for i in order]

        fig, ax = plt.subplots(figsize=(12, max(4, 0.35 * len(names))))
        im = ax.imshow(data, aspect="auto", cmap="viridis")
        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels(names)
        ax.set_xticks(np.arange(timesteps))
        ax.set_xticklabels(np.arange(1, timesteps + 1))
        ax.set_xlabel("Timestep (1 = oldest in window)")
        ax.set_title("Feature x Timestep Importance (Mean |SHAP|)")
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Mean |SHAP value|")
        fig.tight_layout()
        return fig
