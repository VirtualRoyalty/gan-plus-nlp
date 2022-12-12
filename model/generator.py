import torch
import torch.nn as nn

from base import BaseModel


class SimpleGenerator(BaseModel):
    """Generator model class"""

    def __init__(self, input_size: int, output_size: int, dropout_rate: float = 0.2, **kwargs):
        super(SimpleGenerator, self).__init__()
        self.dropout_rate = dropout_rate
        layers = self.get_block(input_size, output_size)
        layers.extend(self.get_block(output_size, output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, noise: torch.Tensor):
        return self.layers(noise)

    def get_block(self, input_size: int, output_size: int):
        return [
            nn.Linear(input_size, output_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.LayerNorm(output_size),
        ]


class ContextualGenerator(BaseModel):
    """Contextual Generator model class with Attention"""

    def __init__(self, input_size: int, output_size: int, dropout_rate: float = 0.2, **kwargs):
        super(ContextualGenerator, self).__init__()
        self.dropout_rate = dropout_rate
        self.attention = nn.MultiheadAttention(embed_dim=input_size, num_heads=1, batch_first=True)
        self.out = nn.Sequential(*self.get_block(input_size, output_size))

    def forward(self, noise: torch.Tensor):
        context_noise = self.attention(query=noise, key=noise, value=noise, need_weights=False)[0]
        return self.out(context_noise)

    def get_block(self, input_size: int, output_size: int):
        return [
            nn.Linear(input_size, output_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.LayerNorm(output_size),
        ]
