import pytest
import torch

from model import SimpleGenerator, ContextualGenerator
from model import DiscriminatorForTokenClassification

from typing import Tuple


@pytest.fixture
def simple_generator_model(input_size: int, output_size: int):
    return SimpleGenerator(input_size, output_size)


@pytest.mark.parametrize(
    "input_size,seq_len,output_size,expected",
    [
        (100, 1, 200, (1, 200)),
        (10, 100, 20, (100, 20)),
        (1, 1, 1, (1, 1)),
    ],
)
def test_simple_generator_forward(
    input_size: int,
    seq_len: int,
    output_size: int,
    expected: Tuple[int],
):
    model = SimpleGenerator(input_size, output_size)
    noise = torch.rand(1, seq_len, input_size)
    ouput = model.forward(noise)
    assert ouput.shape[1:] == expected
