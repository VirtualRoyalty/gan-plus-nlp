import torch

from base import *


class SimpleGenerator(BaseModel):
    """Generator model class"""

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 dropout_rate: float = 0.2,
                 **kwargs):
        super(SimpleGenerator, self).__init__()
        self.dropout_rate = dropout_rate
        layers = self.get_block(input_size, output_size)
        layers.extend(self.get_block(output_size, output_size))
        self.layers = nn.Sequential(layers)

    def forward(self, noise: torch.Tensor):
        return self.layers(noise)

    def get_block(self,
                  input_size: int,
                  output_size: int):
        return [nn.Linear(input_size, output_size),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(self.dropout_rate),
                nn.BatchNorm1d(output_size)]
