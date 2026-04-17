# Code adapted directly from the reproduction of the GPT4TS paper done by liaoyuhua
# Original source: https://github.com/liaoyuhua/GPT-TS
# Credit to the authors of the paper and repository.

import random
import numpy as np
import torch


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Arguments:
            patience (int): number of epochs to wait after the last improvement before stopping training.
            verbose (bool): whether to print out current training and validation results for each epoch.
            delta (float) : minimum reduction in performance error to be considered an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.ins = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        """
        Checks validation loss and perform early stopping if needed.

        Arguments:
            val_loss (float) : current loss observed in validation set
            model (nn.Module): current model weights
            path (str)       : best model's folder save path
        """

        if np.isnan(val_loss):
          print("Observed instability/nan loss")
          self.ins = True
          self.early_stop = True
          return
      
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        """
        Saves current model weights to the best model's save path.

        Arguments:
            val_loss (float) : current loss observed in validation set
            model (nn.Module): current model weights
            path (str)       : best model's folder save path
        """
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), f"{path}/checkpoint.pth")
        self.val_loss_min = val_loss


class StandardScaler:
    """
    Scaler that standardises features by removing the mean and scaling to unit variance.
    """

    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, data):
        """
        Computes mean and std from given training data.
        """

        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        """
        Scales features according to traning mean and std computed.
        """
        mean = (
            torch.from_numpy(self.mean).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.mean
        )
        std = (
            torch.from_numpy(self.std).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.std
        )
        std = np.where(std == 0, 1e-8, std)  # Avoid division by zero
        return (data - mean) / std

    def inverse_transform(self, data):
        """
        Scales features back to their original scales with training mean and std.
        """
        mean = (
            torch.from_numpy(self.mean).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.mean
        )
        std = (
            torch.from_numpy(self.std).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.std
        )
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean


def set_seed(seed):
    """
    Sets seed for the Python standard library `random`, library `numpy` and `pytorch`
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
