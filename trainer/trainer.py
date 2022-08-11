import torch
import functools
import torch.nn as nn
from torch.optim import Adam, AdamW
from typing import List, Dict, Mapping, Optional
from transformers import AutoModel, get_constant_schedule_with_warmup
from transformers.trainer_pt_utils import numpy_pad_and_concatenate

from base import BaseTrainer
from model import compute_metrics, DiscriminatorForTokenClassification, TokenClassifierOutput


class TrainerTokenClassification(BaseTrainer):
    """simple trainer"""

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
        self._define_optimizer()
        self._define_scheduler()
        self.model.to(self.device)

    def training_step(self,
                      batch: Mapping[str, torch.Tensor],
                      log_env: Optional[Dict] = None):
        batch = self._prepare_inputs(batch)
        output = self.model(input_ids=batch['input_ids'],
                            input_mask=batch['attention_mask'],
                            labels=batch['labels'])
        self._train_logging(log_env, output=output)

        self.optimizer.zero_grad()
        output.loss.backward()
        self.optimizer.step()
        if self.config['apply_scheduler']:
            self.scheduler.step()

        epoch_info = dict(loss=output.loss.item())
        return epoch_info

    def train_mode_on(self):
        self.model.train()

    @torch.no_grad()
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
            output = model(input_ids=batch['input_ids'],
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

    @staticmethod
    def _train_logging(log_env: Optional[Mapping] = None,
                       info: Optional[Mapping] = None,
                       output: Optional[TokenClassifierOutput] = None,
                       **kwargs):
        if log_env is not None:
            log_env['train/discriminator_loss'].log(output.loss.item())
        return

    @staticmethod
    def _valid_logging(log_env: Optional[Mapping] = None,
                       info: Optional[Mapping] = None,
                       output: Optional[TokenClassifierOutput] = None,
                       **kwargs):
        if log_env is not None:
            log_env['valid/discriminator_loss'].log(info['loss'])
            log_env['valid/discriminator_accuracy'].log(info['overall_accuracy'])
            log_env['valid/f1'].log(info['overall_f1'])
            log_env['valid/precision'].log(info['overall_precision'])
            log_env['valid/recall'].log(info['overall_recall'])
            log_env['valid/detailed_metrics'].log(info)

    def _define_optimizer(self):
        if self.config['frozen_backbone']:
            self.model.freeze_backbone()
        m_vars = [p for p in self.model.parameters() if p.requires_grad]
        print(f'Trainable layers {len(m_vars)}')
        self.optimizer = AdamW(m_vars, lr=self.config['lr_discriminator'])

    def _define_scheduler(self):
        _default_scheduler = get_constant_schedule_with_warmup
        if self.config['apply_scheduler']:
            train_size = self.config['num_train_examples']
            batch_size = self.config['batch_size']
            epochs = self.config['num_train_epochs']
            self.config['num_train_steps'] = int(train_size / batch_size * epochs)
            self.config['num_warmup_steps_d'] = int(self.config['num_train_steps'] * self.config['warmup_proportion_d'])
            self.scheduler = _default_scheduler(self.optimizer,
                                                num_warmup_steps=self.config['num_warmup_steps_d'])
