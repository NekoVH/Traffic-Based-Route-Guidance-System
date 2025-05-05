import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reference:
# https://www.youtube.com/watch?v=rdz0UqQz5Sw
# https://www.youtube.com/watch?v=0_PgWWmauHk
# https://www.kaggle.com/code/fanbyprinciple/learning-pytorch-3-coding-an-rnn-gru-lstm
class GRU(nn.Module):
    # We can adjust these parameters later when the data processing is working
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(GRU, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.gru(x, h0)

        out = out[:, -1, :]

        return self.fc(out)