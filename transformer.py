import torch
import torch.nn as nn
from pytorch_lightning.demos.transformer import PositionalEncoding
from torch.utils.checkpoint import checkpoint


class Transformer(nn.Module):
    def __init__(self, input_dim=1, model_dim=32, n_heads=4, num_layers=2, output_dim=1, dropout=0.1, use_checkpoint=True):
        super(Transformer, self).__init__()

        self.input_dim = input_dim
        self.model_dim = model_dim
        self.use_checkpoint = use_checkpoint

        self.embedding = nn.Linear(input_dim, model_dim)  # (batch_size, seq_len, input_dim) -> (batch_size, seq_len, model_dim)

        self.positional_encoding = PositionalEncoding(model_dim)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Use pre-norm for better stability
            dim_feedforward=model_dim * 2  # Reduce feedforward dimension
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.fc_out = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        # Ensure the input shape is (batch_size, seq_len, input_dim)
        x = self.embedding(x)

        x = self.positional_encoding(x)

        # Transformer expects (seq_len, batch_size, model_dim)
        x = x.permute(1, 0, 2)

        if self.use_checkpoint and self.training:
            # Use gradient checkpointing during training
            x = checkpoint(self.transformer_encoder, x)
        else:
            x = self.transformer_encoder(x)

        x = x.mean(dim=0)  # Average across the sequence length
        x = self.fc_out(x)

        return x

