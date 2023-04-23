import torch
import functools
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from transformers import AutoModel, get_constant_schedule_with_warmup
from transformers.trainer_pt_utils import numpy_pad_and_concatenate

from typing import List, Dict, Mapping, Optional

from base import BaseTrainer
from model import (
    compute_metrics,
    compute_clf_metrics,
    Discriminator,
    DiscriminatorForSequenceClassification,
    DiscriminatorForTokenClassification,
    ClassifierOutput,
)


class GANTrainerSequenceClassification(BaseTrainer):
    """GAN trainer for sequence classification tasks"""

    def __init__(
        self,
        config: Dict,
        discriminator: DiscriminatorForSequenceClassification,
        generator: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        valid_dataloader: torch.utils.data.DataLoader,
        ce_ignore_index: Optional[int] = -100,
        device=None,
    ):
        self.config = config
        self.model = discriminator
        self.hidden_size = self.model.encoder.config.hidden_size
        self.generator = generator
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.device = device
        self._define_optimizer()
        self._define_scheduler()
        self.model.to(self.device)
        self.generator.to(self.device)

    def training_step(self, batch: Mapping[str, torch.Tensor], log_env: Optional[Dict] = None):
        batch = self._prepare_inputs(batch)
        batch_size = batch["input_ids"].shape[0]
        seq_len = batch["input_ids"].shape[1]

        output = self.model(
            input_ids=batch["input_ids"],
            input_mask=batch["attention_mask"],
            labels=batch["labels"],
            labeled_mask=batch["labeled_mask"],
        )
        noise = torch.rand(batch_size, self.config["noise_size"], device=self.device)
        gen_states = self.generator(noise)

        fake_output = self.model(external_states=gen_states, input_mask=batch["attention_mask"])

        # Generator loss estimation
        cheat_loss = -1 * torch.mean(torch.log(1 - fake_output.probs[:, -1] + self.config["epsilon"]))
        feat_sim_loss = torch.mean(
            torch.pow(
                torch.mean(output.hidden_states, dim=0) - torch.mean(fake_output.hidden_states, dim=0), 2
            )
        )
        generator_loss = cheat_loss + feat_sim_loss

        # Discriminator loss estimation
        unsup_fake_loss = fake_output.loss
        unsup_real_loss = -torch.mean(torch.log(1 - output.probs[:, -1] + self.config["epsilon"]))
        discriminator_loss = output.loss + unsup_fake_loss + unsup_real_loss

        self._train_logging(log_env, dloss=discriminator_loss, gloss=generator_loss)

        self.generator_optimizer.zero_grad()
        self.optimizer.zero_grad()
        generator_loss.backward(retain_graph=True)
        discriminator_loss.backward()
        self.generator_optimizer.step()
        self.optimizer.step()

        if self.config["apply_scheduler"]:
            self.scheduler.step()
            self.generator_scheduler.step()

        epoch_info = dict(loss=discriminator_loss.item(), generator_loss=generator_loss.item())
        return epoch_info

    def train_mode_on(self):
        self.model.train()
        self.generator.train()

    @torch.no_grad()
    def predict(
        self, model: nn.Module, data_loader: torch.utils.data.DataLoader, label_names: List[str]
    ) -> Dict:
        total_loss = 0
        predictions = []
        model.eval()
        for step, batch in enumerate(data_loader):
            batch = self._prepare_inputs(batch)
            output = model(
                input_ids=batch["input_ids"], input_mask=batch["attention_mask"], labels=batch["labels"]
            )
            predictions.append(output.logits.cpu().detach().numpy()[:, :-1])
            total_loss += output.loss
        result = compute_clf_metrics(
            predictions=predictions, labels=data_loader.dataset["labels"], label_names=label_names
        )
        result["loss"] = total_loss / len(data_loader)
        return result

    @staticmethod
    def _train_logging(
        log_env: Optional[Mapping] = None,
        info: Optional[Mapping] = None,
        dloss: torch.FloatTensor = None,
        gloss: torch.FloatTensor = None,
        **kwargs,
    ):
        if log_env is not None:
            log_env["train/discriminator_loss"].log(dloss.item())
            log_env["train/generator_loss"].log(gloss.item())

        return

    @staticmethod
    def _valid_logging(
        log_env: Optional[Mapping] = None,
        info: Optional[Mapping] = None,
        output: Optional[ClassifierOutput] = None,
        **kwargs,
    ):
        if log_env is not None:
            log_env["valid/discriminator_loss"].log(info["loss"])
            log_env["valid/discriminator_accuracy"].log(info["overall_accuracy"])
            log_env["valid/f1"].log(info["overall_f1"])
            log_env["valid/precision"].log(info["overall_precision"])
            log_env["valid/recall"].log(info["overall_recall"])
            log_env["valid/detailed_metrics"].log(info)

    def on_train_start(self):
        train_info = {"total_train_loss": 0, "total_generator_loss": 0}
        return train_info

    def on_train_end(self, info: Dict, verbose: Optional[bool] = True):
        info["total_train_loss"] /= len(self.train_dataloader)
        info["total_generator_loss"] /= len(self.train_dataloader)
        if verbose:
            print(f"\tTrain loss discriminator: {info['total_train_loss']:.3f}")
            print(f"\tTrain loss generator: {info['total_generator_loss']:.3f}")

    def on_epoch_end(self, **kwargs):
        kwargs["train_info"]["total_train_loss"] += kwargs["epoch_info"]["loss"]
        kwargs["train_info"]["total_generator_loss"] += kwargs["epoch_info"]["generator_loss"]

    def _define_optimizer(self):
        # discriminator optimizer
        if self.config["frozen_backbone"]:
            self.model.freeze_backbone()
        m_vars = [p for p in self.model.parameters() if p.requires_grad]
        print(f"Trainable layers {len(m_vars)}")
        self.optimizer = AdamW(m_vars, lr=self.config["lr_discriminator"])
        # generator optimizer
        g_vars = [v for v in self.generator.parameters()]
        self.generator_optimizer = torch.optim.AdamW(g_vars, lr=self.config["lr_generator"])

    def _define_scheduler(self):
        _default_scheduler = get_constant_schedule_with_warmup
        if self.config["apply_scheduler"]:
            # discriminator scheduler
            train_size = self.config["num_train_examples"]
            batch_size = self.config["batch_size"]
            epochs = self.config["num_train_epochs"]
            self.config["num_train_steps"] = int(train_size / batch_size * epochs)
            self.config["num_warmup_steps_d"] = int(
                self.config["num_train_steps"] * self.config["warmup_proportion_d"]
            )
            self.scheduler = _default_scheduler(
                self.optimizer, num_warmup_steps=self.config["num_warmup_steps_d"]
            )
            # generator scheduler
            self.config["num_warmup_steps_g"] = int(
                self.config["num_train_steps"] * self.config["warmup_proportion_g"]
            )
            self.generator_scheduler = _default_scheduler(
                self.generator_optimizer, num_warmup_steps=self.config["num_warmup_steps_g"]
            )


class GANTrainerTokenClassification(BaseTrainer):
    """GAN trainer for token classification tasks"""

    def __init__(
        self,
        config: Dict,
        discriminator: DiscriminatorForTokenClassification,
        generator: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        valid_dataloader: torch.utils.data.DataLoader,
        ce_ignore_index: Optional[int] = -100,
        device=None,
    ):
        self.config = config
        self.model = discriminator
        self.hidden_size = self.model.encoder.config.hidden_size
        self.generator = generator
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.device = device
        self._define_optimizer()
        self._define_scheduler()
        self.model.to(self.device)
        self.generator.to(self.device)

    def training_step(self, batch: Mapping[str, torch.Tensor], log_env: Optional[Dict] = None):
        batch = self._prepare_inputs(batch)
        batch_size = batch["input_ids"].shape[0]
        seq_len = batch["input_ids"].shape[1]

        output = self.model(
            input_ids=batch["input_ids"],
            input_mask=batch["attention_mask"],
            labels=batch["labels"],
            labeled_mask=batch["labeled_mask"],
        )
        if self.config["GAN_TYPE"] != "mixed":
            noise = torch.rand(batch_size, seq_len, self.config["noise_size"], device=self.device)
            gen_states = self.generator(noise)
        else:
            mixed_fake_ratio = self.config.get("mixed_fake_ratio", 0.3)
            noise_len = int(seq_len * mixed_fake_ratio)
            noise = torch.rand(batch_size, noise_len, self.config["noise_size"], device=self.device)
            hstates_len = seq_len - noise_len
            rand_indexes = torch.randperm(seq_len)[:hstates_len]
            selected_hiden_states = output.hidden_states[:, rand_indexes, :]
            gen_states = self.generator(noise, real_encoded_samples=selected_hiden_states)
        fake_output = self.model(external_states=gen_states, input_mask=batch["attention_mask"])

        # Generator loss estimation
        cheat_loss = -1 * torch.mean(torch.log(1 - fake_output.probs[:, -1] + self.config["epsilon"]))
        feat_sim_loss = torch.mean(
            torch.pow(
                torch.mean(output.hidden_states, dim=0) - torch.mean(fake_output.hidden_states, dim=0), 2
            )
        )
        generator_loss = cheat_loss + feat_sim_loss

        # Discriminator loss estimation
        unsup_fake_loss = fake_output.loss
        unsup_real_loss = -torch.mean(torch.log(1 - output.probs[:, -1] + self.config["epsilon"]))
        discriminator_loss = output.loss + unsup_fake_loss + unsup_real_loss

        self._train_logging(log_env, dloss=discriminator_loss, gloss=generator_loss)

        self.generator_optimizer.zero_grad()
        self.optimizer.zero_grad()
        generator_loss.backward(retain_graph=True)
        discriminator_loss.backward()
        self.generator_optimizer.step()
        self.optimizer.step()

        if self.config["apply_scheduler"]:
            self.scheduler.step()
            self.generator_scheduler.step()

        epoch_info = dict(loss=discriminator_loss.item(), generator_loss=generator_loss.item())
        return epoch_info

    def train_mode_on(self):
        self.model.train()
        self.generator.train()

    @torch.no_grad()
    def predict(
        self, model: nn.Module, data_loader: torch.utils.data.DataLoader, label_names: List[str]
    ) -> Dict:
        total_loss = 0
        predictions = []
        model.eval()
        for step, batch in enumerate(data_loader):
            batch = self._prepare_inputs(batch)
            output = model(
                input_ids=batch["input_ids"], input_mask=batch["attention_mask"], labels=batch["labels"]
            )
            predictions.append(output.logits.cpu().detach().numpy()[:, :, :-1])
            total_loss += output.loss
        predictions = functools.reduce(numpy_pad_and_concatenate, predictions)
        result = compute_metrics(
            predictions=predictions, labels=data_loader.dataset["labels"], label_names=label_names
        )
        result["loss"] = total_loss / len(data_loader)
        return result

    @staticmethod
    def _train_logging(
        log_env: Optional[Mapping] = None,
        info: Optional[Mapping] = None,
        dloss: torch.FloatTensor = None,
        gloss: torch.FloatTensor = None,
        **kwargs,
    ):
        if log_env is not None:
            log_env["train/discriminator_loss"].log(dloss.item())
            log_env["train/generator_loss"].log(gloss.item())

        return

    @staticmethod
    def _valid_logging(
        log_env: Optional[Mapping] = None,
        info: Optional[Mapping] = None,
        output: Optional[ClassifierOutput] = None,
        **kwargs,
    ):
        if log_env is not None:
            log_env["valid/discriminator_loss"].log(info["loss"])
            log_env["valid/discriminator_accuracy"].log(info["overall_accuracy"])
            log_env["valid/f1"].log(info["overall_f1"])
            log_env["valid/precision"].log(info["overall_precision"])
            log_env["valid/recall"].log(info["overall_recall"])
            log_env["valid/detailed_metrics"].log(info)

    def on_train_start(self):
        train_info = {"total_train_loss": 0, "total_generator_loss": 0}
        return train_info

    def on_train_end(self, info: Dict, verbose: Optional[bool] = True):
        info["total_train_loss"] /= len(self.train_dataloader)
        info["total_generator_loss"] /= len(self.train_dataloader)
        if verbose:
            print(f"\tTrain loss discriminator: {info['total_train_loss']:.3f}")
            print(f"\tTrain loss generator: {info['total_generator_loss']:.3f}")

    def on_epoch_end(self, **kwargs):
        kwargs["train_info"]["total_train_loss"] += kwargs["epoch_info"]["loss"]
        kwargs["train_info"]["total_generator_loss"] += kwargs["epoch_info"]["generator_loss"]

    def _define_optimizer(self):
        # discriminator optimizer
        if self.config["frozen_backbone"]:
            self.model.freeze_backbone()
        m_vars = [p for p in self.model.parameters() if p.requires_grad]
        print(f"Trainable layers {len(m_vars)}")
        self.optimizer = AdamW(m_vars, lr=self.config["lr_discriminator"])
        # generator optimizer
        g_vars = [v for v in self.generator.parameters()]
        self.generator_optimizer = torch.optim.AdamW(g_vars, lr=self.config["lr_generator"])

    def _define_scheduler(self):
        _default_scheduler = get_constant_schedule_with_warmup
        if self.config["apply_scheduler"]:
            # discriminator scheduler
            train_size = self.config["num_train_examples"]
            batch_size = self.config["batch_size"]
            epochs = self.config["num_train_epochs"]
            self.config["num_train_steps"] = int(train_size / batch_size * epochs)
            self.config["num_warmup_steps_d"] = int(
                self.config["num_train_steps"] * self.config["warmup_proportion_d"]
            )
            self.scheduler = _default_scheduler(
                self.optimizer, num_warmup_steps=self.config["num_warmup_steps_d"]
            )
            # generator scheduler
            self.config["num_warmup_steps_g"] = int(
                self.config["num_train_steps"] * self.config["warmup_proportion_g"]
            )
            self.generator_scheduler = _default_scheduler(
                self.generator_optimizer, num_warmup_steps=self.config["num_warmup_steps_g"]
            )
