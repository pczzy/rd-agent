import torch
import torch.nn as nn

class BidirectionalGRUAttentionTimeSeriesModel(nn.Module):
    def __init__(self, num_features: int, num_timesteps: int = None):
        super().__init__()
        self.input_projection = nn.Linear(num_features, 128)
        self.projection_layernorm = nn.LayerNorm(128)
        self.projection_activation = nn.GELU()
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            batch_first=True,
        )
        self.query = nn.Parameter(torch.randn(1, 256))
        self.layer_norm = nn.LayerNorm(256)
        self.mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.projection_activation(self.projection_layernorm(self.input_projection(x)))
        gru_out, _ = self.gru(projected)
        batch_size = gru_out.size(0)
        query = self.query.unsqueeze(0).expand(batch_size, -1, -1)
        attn_out, _ = self.attention(query, gru_out, gru_out)
        attn_out = attn_out.squeeze(1)
        residual = gru_out.mean(dim=1)
        z = self.layer_norm(attn_out + residual)
        output = self.mlp(z)
        return output

model_cls = BidirectionalGRUAttentionTimeSeriesModel