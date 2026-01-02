import matplotlib.pyplot as plt


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
    def plot_accuracy(history: dict) -> plt.Figure:
        """Creates and displays a pie chart showing accuracy distribution.

        Args:
            history: Dictionary containing training history with 'accuracy' values

        Returns:
            matplotlib.figure.Figure: The generated pie chart figure
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        accuracy = history.get('accuracy', 0)
        error = 100 - accuracy

        ax.pie([accuracy, error],
               labels=['Accurate', 'Error'],
               autopct='%1.1f%%',
               colors=['lightgreen', 'lightcoral'])

        ax.set_title('Model Accuracy Distribution')

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
