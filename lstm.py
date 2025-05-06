import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size=7, hidden_size=64, num_layers=1, output_dim=1, dropout_prob=0.2):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, X):
        hidden_states = torch.zeros(self.num_layers, X.size(0), self.hidden_size).requires_grad_()
        cell_states = torch.zeros(self.num_layers, X.size(0), self.hidden_size).requires_grad_()
        out, (hn, cn) = self.lstm(X, (hidden_states.detach(), cell_states.detach()))
        out = out[:, -1, :]
        out = self.fc(out)
        return out
