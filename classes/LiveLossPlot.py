import matplotlib.pyplot as plt
from IPython.display import display, HTML


class LiveLossPlot:
    def __init__(self, update_every: int = 10):
        self.update_every = update_every
        self.train = []
        self.val = []
        self.fig = None
        self.ax = None
        self.handle = None  # display handle

    def reset(self, title: str = ""):
        self.train.clear()
        self.val.clear()

        # Create a NEW figure per run (avoids "closed/vanishing" issues)
        self.fig, self.ax = plt.subplots()
        self.ax.set_title(title)
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")

        # Create a persistent display slot we can update
        self.handle = display(self.fig, display_id=True)

    def update(self, epoch: int, train_loss: float, val_loss: float):
        self.train.append(train_loss)
        self.val.append(val_loss)

        if epoch % self.update_every != 0:
            return

        self.ax.clear()
        self.ax.plot(self.train, label="train")
        self.ax.plot(self.val, label="val")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.legend()

        # Important: redraw + update the same output slot
        self.fig.canvas.draw_idle()
        self.handle.update(self.fig)

    def close(self):

        if self.handle is not None:
            # Replace only this display slot so other cell output remains.
            self.handle.update(HTML(""))
            self.handle = None
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
