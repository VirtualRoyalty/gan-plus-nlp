import torch
import functools
from torch.optim import Adam, AdamW
from typing import Dict, Tuple, Mapping, Union, Any
from transformers import AutoModel, get_constant_schedule_with_warmup
from transformers.trainer_pt_utils import numpy_pad_and_concatenate

from model import *
from model import compute_metrics


class TrainerTokenClassification:
    def __init__(self,
                 config: Dict,
                 discriminator: DiscriminatorForTokenClassification,
                 train_dataloader: torch.utils.data.DataLoader,
                 valid_dataloader: torch.utils.data.DataLoader,
                 ce_ignore_index: Optional[int] = -100,
                 device=None):

        self.config = config
        self.model = discriminator
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.device = device
        self.valid_nll = torch.nn.CrossEntropyLoss(ignore_index=ce_ignore_index)

        # define trainable parameters and optimizer
        if config['frozen_backbone']:
            self.model.freeze_backbone()
        m_vars = [p for p in self.model.parameters() if p.requires_grad]
        print(f'Trainable layers {len(m_vars)}')
        self.optimizer = AdamW(m_vars, lr=config['learning_rate_discriminator'])

        # define scheduler
        if config['apply_scheduler']:
            self._define_scheduler()

        # move model to device
        self.model.to(self.device)

    def train_epoch(self, log_env: Optional[Dict] = None) -> float:
        total_loss = 0
        self.model.train()

        for step, batch in enumerate(self.train_dataloader):
            batch = self._prepare_inputs(batch)
            output = self.model(input_ids=batch['input_ids'].to(self.device),
                                input_mask=batch['attention_mask'],
                                labels=batch['labels'])
            if log_env:
                log_env['train/discriminator_loss'].log(output.loss.item())

            self.optimizer.zero_grad()
            output.loss.backward()
            self.optimizer.step()
            total_loss += output.loss.item()
            if self.config['apply_scheduler']:
                self.scheduler.step()
        return total_loss

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

        if log_env is not None:
            log_env['valid/discriminator_loss'].log(result_metrics['loss'])
            log_env['valid/discriminator_accuracy'].log(result_metrics['overall_accuracy'])
            log_env['valid/f1'].log(result_metrics['overall_f1'])
            log_env['valid/precision'].log(result_metrics['overall_precision'])
            log_env['valid/recall'].log(result_metrics['overall_recall'])
            log_env['valid/detailed_metrics'].log(result_metrics)

        if verbose:
            print(f"\tTrain loss discriminator: {train_loss / len(self.train_dataloader):.3f}")
            print(f"  Test loss discriminator: {result_metrics['loss']:.3f}")
            print(f"  Test accuracy discriminator: {result_metrics['overall_accuracy']:.3f}")
            print(f"  Test f1 discriminator: {result_metrics['overall_f1']:.3f}")
        return result_metrics

    def predict(self,
                model: nn.Module,
                data_loader: torch.utils.data.DataLoader,
                label_names: List[str]
                ) -> Dict:
        total_loss = 0
        predictions = []
        model.eval()
        for step, batch in enumerate(data_loader):
            batch = self._prepare_inputs(batch)
            output = model(input_ids=batch['input_ids'].to(self.device),
                           input_mask=batch['attention_mask'],
                           labels=batch['labels'])
            predictions.append(output.logits.cpu().detach().numpy())
            total_loss += output.loss
        predictions = functools.reduce(numpy_pad_and_concatenate, predictions)
        result = compute_metrics(predictions=predictions,
                                 labels=data_loader.dataset['labels'],
                                 label_names=label_names)
        result['loss'] = total_loss / len(data_loader)
        return result

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

    def _define_scheduler(self):

        _default_scheduler = get_constant_schedule_with_warmup
        train_size = self.config['num_train_examples']
        batch_size = self.config['batch_size']
        epochs = self.config['num_train_epochs']
        self.config['num_train_steps'] = int(train_size / batch_size * epochs)
        self.config['num_warmup_steps_d'] = int(train_size * self.config['warmup_proportion_d'])
        self.scheduler = _default_scheduler(self.optimizer,
                                            num_warmup_steps=self.config['num_warmup_steps_d'])
