import os.path
from model import efficientnet_b0
import matplotlib.pyplot as plt
import torch
from pathlib import Path

def plot_loss_curves(results):
    """Plots training curves of a results' dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["val_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["val_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig('Loss_Accuracy.png')
    plt.show()


class EarlyStopping():
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.early_stopping = False
        self.best_loss = None

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            if self.counter > 0:
                print('Reset early stopping counter from {} to {} !!!'.format(self.counter, 0))
                self.counter = 0
        else:
            self.counter += 1
            print('Early stopping counter {} / {} !!!'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stopping = True


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(target_dir_path, model_name)
    print(f"[INFO] Saving model to: {model_path}")
    torch.save(model.state_dict(), model_path)
