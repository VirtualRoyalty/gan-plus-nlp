import sys
import pytest

from base import BaseModel, BaseTrainer


def test_base_model_child_forward_bad():
    class BaseModelChild(BaseModel):
        pass

    with pytest.raises(TypeError):
        model = BaseModelChild()
        model.forward([1, 2, 3])


def test_base_model_child_forward_good():
    class BaseModelChild(BaseModel):
        def forward(self, inputs):
            return sum(inputs)

    model = BaseModelChild()
    assert 6 == model.forward([1, 2, 3])


def test_base_trainer_child_init():
    class BaseTrainerChild(BaseTrainer):
        pass

    with pytest.raises(TypeError):
        trainer = BaseTrainerChild()


def test_base_trainer_child_predict():
    class BaseTrainerChild(BaseTrainer):
        def _train_logging(self):
            pass

        def _valid_logging(self):
            pass

        def predict(self, inputs):
            return sum(inputs)

        def train_mode_on(self):
            pass

        def training_step(self):
            pass

    trainer = BaseTrainerChild()
    assert 6 == trainer.predict([1, 2, 3])
