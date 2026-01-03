import matplotlib.pyplot as plt
import numpy as np


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
        ax.plot(x, naive, label="Naive (No-Change)")

        ax.set_xlabel("Date")
        ax.set_ylabel("Equity")
        ax.set_title("Equity Curve Comparison")
        ax.legend()

        plt.show()
        return fig
