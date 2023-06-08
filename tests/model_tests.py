import pytest
import torch
import random

from base import BaseModel
from model import SimpleTokenGenerator, ContextualTokenGenerator
from model import DiscriminatorForTokenClassification
from model import ClassifierOutput

from typing import Tuple


@pytest.fixture(params=[SimpleTokenGenerator, ContextualTokenGenerator])
def get_generator_class(request):
    return request.param


@pytest.mark.parametrize(
    "input_size, seq_len, output_size, expected",
    [
        (100, 1, 200, (1, 200)),
        (10, 100, 20, (100, 20)),
        (1, 1, 1, (1, 1)),
    ],
)
def test_generator_forward(
    get_generator_class,
    input_size: int,
    seq_len: int,
    output_size: int,
    expected: Tuple[int],
):
    model = get_generator_class(input_size, output_size)
    noise = torch.rand(1, seq_len, input_size)
    ouput = model.forward(noise)
    assert ouput.shape[1:] == expected


@pytest.fixture(params=["distilbert-base-uncased", "google/electra-small-discriminator"])
def get_discriminator(request):
    print(f"get_discriminator will load {request.param}...")
    return DiscriminatorForTokenClassification(
        encoder_name=request.param, num_labels=random.randint(1, 50)
    )


def test_discriminator_freeze(get_discriminator):
    model = get_discriminator
    model.freeze_backbone()
    for _, parameter in model.encoder.named_parameters():
        assert parameter.requires_grad is False


def test_discriminator_bad_forward(get_discriminator):
    model = get_discriminator
    with pytest.raises(AssertionError):
        model.forward()


@pytest.mark.parametrize(
    "batch_size, seq_len",
    [
        (1, 10),
        (8, 32),
    ],
)
def test_discriminator_good_forward_external_states(
    get_discriminator, get_generator_class, batch_size: int, seq_len: int
):
    model = get_discriminator
    generator = get_generator_class(100, model.encoder.config.hidden_size)
    noise = torch.rand(batch_size, seq_len, 100)
    generator_output = generator.forward(noise)
    output = model.forward(external_states=generator_output)
    assert isinstance(output, ClassifierOutput)
