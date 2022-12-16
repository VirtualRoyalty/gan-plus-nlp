import pytest
import torch

from base import BaseModel
from model import SimpleGenerator, ContextualGenerator
from model import DiscriminatorForTokenClassification

from typing import Tuple


@pytest.mark.parametrize(
    "generator, input_size, seq_len, output_size, expected",
    [
        (SimpleGenerator, 100, 1, 200, (1, 200)),
        (SimpleGenerator, 10, 100, 20, (100, 20)),
        (SimpleGenerator, 1, 1, 1, (1, 1)),
        (ContextualGenerator, 100, 1, 200, (1, 200)),
        (ContextualGenerator, 10, 100, 20, (100, 20)),
        (ContextualGenerator, 1, 1, 1, (1, 1)),
    ],
)
def test_simple_generator_forward(
    generator: BaseModel,
    input_size: int,
    seq_len: int,
    output_size: int,
    expected: Tuple[int],
):
    model = generator(input_size, output_size)
    noise = torch.rand(1, seq_len, input_size)
    ouput = model.forward(noise)
    assert ouput.shape[1:] == expected
