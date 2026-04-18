import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    Multi-layer LSTM baseline for volatility forecasting.
 
    Architecture:
        1. Permute input from (B, N, L) to (B, L, N) for LSTM convention
        2. Process through num_layers LSTM layers with dropout
        3. Take the last time step's hidden state
        4. Project to pred_len via a linear layer
        5. Reshape to (B, 1, pred_len) to match EquityBERT output shape
    """

    def __init__(self, num_series, hidden_size, num_layers, pred_len, dropout=0.2):
        """
        Arguments:
            num_series (int)  : number of input features N
            hidden_size (int) : LSTM hidden dimension
            num_layers (int)  : number of stacked LSTM layers
            pred_len (int)    : forecast horizon P
            dropout (float)   : dropout between LSTM layers (default: 0.2)
        """

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=num_series,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0  # No dropout if only 1 layer
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, pred_len)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        """
        Forward pass.
 
        Arguments:
            x : (B, num_series, seq_len) input tensor
 
        Returns:
            (B, 1, pred_len) forecast tensor
        """
        # LSTM expects (B, seq_len, input_size)
        x = x.permute(0, 2, 1)  # → (B, seq_len, num_series)

        out, _ = self.lstm(x) # out: (B, seq_len, hidden_size)
        
        last_hidden = out[:, -1, :] # Take the last time step's hidden state → (B, hidden_size)
        last_hidden = self.dropout(last_hidden) # Apply dropout
        out = self.fc(last_hidden) # → (B, pred_len)


        return out.unsqueeze(1)  # (B, 1, pred_len)
    
    @property
    def num_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}