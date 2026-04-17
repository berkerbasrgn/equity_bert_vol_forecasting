import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, num_series, hidden_size, num_layers, pred_len, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=num_series,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        # x: (B, num_series, seq_len)
        x = x.permute(0, 2, 1)  # → (B, seq_len, num_series)

        out, _ = self.lstm(x)

        last_hidden = out[:, -1, :]

        out = self.fc(last_hidden)


        return out.unsqueeze(1)  # (B, 1, pred_len)
    
    @property
    def num_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}