import torch
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Mapping, Union, Any, Optional


class BaseTrainer(ABC):
    """
    Base class for all trainers
    """

    def train_epoch(self, log_env: Optional[Dict] = None) -> float:
        total_loss = 0
        self.model.train()
        for step, batch in enumerate(self.train_dataloader):
            loss = self.training_step(batch, log_env)
            total_loss += loss
        return total_loss / len(self.train_dataloader)

    @abstractmethod
    def training_step(self):
        """
        Training step logic
        """
        return NotImplementedError

    @torch.no_grad()
    def validation(self,
                   train_loss: float,
                   verbose: Optional[bool] = True,
                   log_env: Optional[Dict] = None,
                   **kwargs
                   ) -> Dict:

        result_metrics = self.predict(model=self.model,
                                      data_loader=self.valid_dataloader,
                                      label_names=self.config['label_names'])
        self._valid_logging(log_env, info=result_metrics)

        if verbose:
            print(f"\tTrain loss discriminator: {train_loss:.3f}")
            print(f"\tTest loss discriminator: {result_metrics['loss']:.3f}")
            print(f"\tTest accuracy discriminator: {result_metrics['overall_accuracy']:.3f}")
            print(f"\tTest f1 discriminator: {result_metrics['overall_f1']:.3f}")
        return result_metrics

    @abstractmethod
    def predict(self):
        """
        Training step logic
        """
        return NotImplementedError

    @abstractmethod
    def _train_logging(self):
        """
        Training step logic
        """
        return NotImplementedError

    @abstractmethod
    def _valid_logging(self):
        """
        Training step logic
        """
        return NotImplementedError

    def _prepare_inputs(self,
                        data: Union[torch.Tensor, Any]
                        ) -> Union[torch.Tensor, Any]:
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_inputs(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_inputs(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = dict(device=self.device)
            return data.to(**kwargs)
        return data
