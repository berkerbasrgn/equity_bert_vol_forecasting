import time
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader, Dataset
from torch.optim import Adam
from torch.optim import lr_scheduler
from .utils import EarlyStopping
from .loss import mae_loss
import os

class Trainer:
    """
    Trainer for model training and validation with early stopping. This extended trainer is designed
    specifically for models like Vola-BERT where input is a Python tuple containing multiple PyTorch tensors.

    Default loss function is MAE.
    """

    def __init__(
        self,
        model: nn.Module,
        use_amp: bool = False,
        features: str = "M",
        inverse: bool = False,
        num_workers: int = 0,
        save_path="checkpoint",
        patience=10,
        verbose: bool = True,
        checkpoint_file="checkpoint.pth",
    ) -> None:
        """
        Arguments:
            model (nn.Module): the model to train
            use_amp (bool)   : whether to use automatic mixed precision training
            features (str)   : the features to use, either S (univariate predict univariate)
                                 or MS (multivariate predict univariate)
            inverse (bool)   : whether to inverse the target sequence
            num_workers (int): number of workers for data loader
            save_path (str)  : best model's folder save path
            patience (int)   : number of epochs to wait after the last improvement before stopping training
            verbose (bool)   : whether to report traning performance during each epoch
        """
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"Using device: {self.device}")


        self.model = model.to(self.device)
        self.use_amp = use_amp  # automatic mixed precision training (doesn't work with MPS, so we will just ignore it if MPS is available)

        self.early_stopping = EarlyStopping(patience=patience, verbose=verbose)

        self.f_dim = -1 if features == "MS" else 0  # -1 for MS, 0 for M and S
        self.inverse = inverse
        self.num_workers = num_workers
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            print(f"Path {self.save_path} not exists, creating...")
            os.makedirs(self.save_path)

        self.verbose = verbose
        self.checkpoint_file = os.path.join(self.save_path, "model.pth")
      
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        batch_size: int,
        max_epochs: int,
        lr: float,
        pct_start: float = 0.3
    ):
        """
        Runs a full epoch training loop with early stopping using given datasets.
        
        Arguments:
            train_dataset (Dataset): dataset used for training
            val_dataset (Dataset)  : dataset used for validation and early stopping
            batch_size (int)       : number of samples per training and validation batch.
            max_epochs (int)       : maximum number of training epochs.
            lr (float)             : initial learning rate.
            pct_start (float)      : deprecated, not used.
        """

        # save meta data
        self.train_info = {
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "lr": lr,
            "pct_start": pct_start,
            "num_workers": self.num_workers,
            "use_amp": self.use_amp,
        }
        self.max_epochs = max_epochs

        # initialize data loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        train_steps = len(train_loader)
        self.train_info.update({"train_steps": train_steps})

        # initialize optimizer
        self.optimizer = Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr
        )

        # initialize lr scheduler
        self.scheduler = lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            steps_per_epoch=train_steps,
            pct_start=pct_start,
            epochs=max_epochs,
            max_lr=lr,
        )

        scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        train_mae_list = []
        train_mse_list = []
        val_mae_list = []
        val_mse_list = []

        for epoch in range(max_epochs):
            start_time = time.time()
            train_mae, train_mse = self._train_epoch(train_loader, scaler)
            val_mae, val_mse = self._val_epoch(val_loader)

            train_mae_list.append(train_mae)
            val_mae_list.append(val_mae)
            train_mse_list.append(train_mse)
            val_mse_list.append(val_mse)

            self.early_stopping(val_mse, self.model, self.save_path)

            if self.early_stopping.early_stop:
                if self.verbose:
                    print("Early stopping")
                break
              
            if self.verbose:
                print(
                    f"Epoch: {epoch+1}/{self.max_epochs} | Train MAE: {train_mae} | Val MAE: {val_mae} |"
                      f" Train MSE: {train_mse} | Val MSE: {val_mse} | Time: {time.time()-start_time: .3f}s"
                )
        return train_mae_list, val_mae_list, train_mse_list, val_mse_list

    def input_to_device(self, x: tuple):
        """
        Moves all input tensors to used device.
        Handles both BERT (with tokens dict) and LSTM (tensor only) models.
        """
        
        # Check if x[1] is a dictionary (BERT tokens) or a tensor (LSTM)
        if isinstance(x[1], dict):
            # BERT model: (seq_x, tokens_dict)
            return tuple([x[0].to(self.device), {tok_name: tok_val.to(self.device) for tok_name, tok_val in x[1].items()}])
        else:
            # LSTM model: just return the tensor x[0]
            return x[0].to(self.device)
        

    def _train_epoch(self, train_loader, scaler=None):
        """
        Runs one training epoch with optionally mixed precision with a GradScaler.
        """

        self.model.train()

        total_loss = 0
        total_mse = 0
        for x, y in train_loader:

          
            x = self.input_to_device(x)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            pred, y_true = self._run_on_batch(train_loader.dataset, x, y)

            loss = mae_loss(pred, y_true)
            mse = torch.mean((pred - y_true) ** 2)

            if self.use_amp:
                scaler.scale(loss).backward()

                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), max_norm=1.0)
              
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.item()
            total_mse += mse.item()

        # Save checkpoint after each epoch
        torch.save(self.model.state_dict(), self.checkpoint_file)
        
        return total_loss / len(train_loader), total_mse / len(train_loader)
    

    def _val_epoch(self, val_loader):
        """
        Evaluates model's current performance on validation set.        
        """

        self.model.eval()

        total_loss = 0
        total_mse = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = self.input_to_device(x)
                y = y.to(self.device)

                pred, y_true = self._run_on_batch(val_loader.dataset, x, y)

                loss = mae_loss(pred, y_true)
                mse = torch.mean((pred - y_true) ** 2)

                total_loss += loss.item()
                total_mse += mse.item()

        return total_loss / len(val_loader), total_mse / len(val_loader)

    def _run_on_batch(
        self,
        dataset_object: Dataset,
        x_batch: tuple,
        y_batch: torch.Tensor = None,
    ):
        """
        Computes model's output based on given input batch.
        """

        if self.use_amp and self.device.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = self.model(x_batch)
        else:
            outputs = self.model(x_batch)

        if self.inverse:
            outputs = dataset_object.inverse_transform(outputs)

        if y_batch is not None:
            y_batch = y_batch[:, self.f_dim :, :].to(self.device)  # B, N (target), Y

        outputs = outputs[:, self.f_dim :, :]  # B, N (target), Y

    

        return outputs, y_batch

    def test(self, test_dataset, batch_size):
        """
        Evaluates model's MAE performance on given test datset.
        """

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        self.model.load_state_dict(torch.load(self.checkpoint_file))

        self.model.eval()

        total_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = self.input_to_device(x)
                y = y.to(self.device)

                pred, y_true = self._run_on_batch(test_loader.dataset, x, y)

                loss = mae_loss(pred, y_true)

                total_loss += loss.item()

        return total_loss / len(test_loader)