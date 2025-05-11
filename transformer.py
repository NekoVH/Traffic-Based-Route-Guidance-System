import torch
import torch.nn as nn
from pytorch_lightning.demos.transformer import PositionalEncoding


class Transformer(nn.Module):
    def __init__(self, input_dim=1, model_dim=64, n_heads=4, num_layers=2, output_dim=1, dropout=0.1):
        super(Transformer, self).__init__()

        self.input_dim = input_dim
        self.model_dim = model_dim

        # Replace embedding with a linear layer for numerical input
        self.embedding = nn.Linear(input_dim, model_dim)  # (batch_size, seq_len, input_dim) -> (batch_size, seq_len, model_dim)

        self.positional_encoding = PositionalEncoding(model_dim)

        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=n_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.fc_out = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        # Ensure the input shape is (batch_size, seq_len, input_dim)
        x = self.embedding(x)

        x = self.positional_encoding(x)

        # Transformer expects (seq_len, batch_size, model_dim)
        x = x.permute(1, 0, 2)

        x = self.transformer_encoder(x)

        x = x.mean(dim=0)  # Average across the sequence length
        x = self.fc_out(x)

        return x

