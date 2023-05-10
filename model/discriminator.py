import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModel, AutoTokenizer, AutoConfig

from typing import Optional, Tuple, Dict

from base import BaseModel
from model.utils import ClassifierOutput, NewClassifierOutput


class Discriminator(BaseModel):
    def get_tokenizer(self) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(self.encoder_name)

    def freeze_backbone(self) -> None:
        for name, parameter in self.encoder.named_parameters():
            parameter.requires_grad = False


class DiscriminatorForSequenceClassification(Discriminator):
    """Discriminator model for sequence classification tasks with transformer backbone"""

    def __init__(
        self,
        encoder_name: str,
        num_labels: int = 10,
        dropout_rate: Optional[float] = 0.15,
        ce_ignore_index: Optional[int] = -100,
        epsilon: Optional[float] = 1e-8,
        gan_training: bool = False,
        **kwargs,
    ):
        super(DiscriminatorForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.encoder_name = encoder_name
        self.encoder = AutoModel.from_pretrained(encoder_name)
        classifier_dropout = (
            self.encoder.config.classifier_dropout
            if hasattr(self.encoder.config, "classifier_dropout")
            else None
        )
        self.dropout = nn.Dropout(dropout_rate if classifier_dropout is None else classifier_dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.softmax = nn.Softmax(dim=-1)
        self.loss_fct = CrossEntropyLoss(ignore_index=ce_ignore_index)
        self.epsilon = epsilon
        self.gan_training = gan_training
        if self.gan_training:
            print("Training with GAN mode on!")
            self.real_labels = torch.arange(num_labels) != (num_labels - 1)
            self.fake_index = -1
            print(f"Default fake label index is {self.fake_index}")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_mask: Optional[torch.Tensor] = None,
        external_states: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        labeled_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> ClassifierOutput:
        # simple check
        if input_ids is None and external_states is None:
            raise AssertionError("Empty input: input_ids and external states are empty")

        if input_ids is not None:
            outputs = self.encoder(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
            sequence_output = outputs.last_hidden_state[:, 0]  # get CLS embedding

            # add generator input to hidden states
            if external_states is not None:
                sequence_output = torch.cat([sequence_output, external_states], dim=0)
        else:
            sequence_output = external_states

        sequence_output_drop = self.dropout(sequence_output)
        logits = self.classifier(sequence_output_drop)
        probs = self.softmax(logits)

        loss = self.compute_loss(logits=logits, probs=probs, labels=labels, labeled_mask=labeled_mask)

        return ClassifierOutput(loss=loss, logits=logits, probs=probs, hidden_states=sequence_output)

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        labeled_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.FloatTensor]:
        loss = None
        if labels is not None:
            if labeled_mask is not None:
                labeled_mask = labeled_mask.bool()
                logits = logits[labeled_mask]
                labels = labels[labeled_mask]
                if logits.shape[0] == 0:
                    return torch.FloatTensor([0]).cuda()

            if self.gan_training:
                logits = logits[:, self.real_labels]

            loss = self.loss_fct(logits, labels.view(-1))
        elif self.gan_training:
            loss = -torch.mean(torch.log(probs[:, self.fake_index] + self.epsilon))
        return loss


class DiscriminatorForMultiLabelClassification(Discriminator):
    """Discriminator model for sequence classification tasks with transformer backbone"""

    def __init__(
        self,
        encoder_name: str,
        num_labels: int = 4,
        dropout_rate: Optional[float] = 0.15,
        ce_ignore_index: Optional[int] = -100,
        epsilon: Optional[float] = 1e-8,
        gan_training: bool = False,
        **kwargs,
    ):
        super(DiscriminatorForMultiLabelClassification, self).__init__()
        self.num_labels = num_labels
        self.encoder_name = encoder_name
        self.encoder = AutoModel.from_pretrained(encoder_name)
        classifier_dropout = (
            self.encoder.config.classifier_dropout
            if hasattr(self.encoder.config, "classifier_dropout")
            else None
        )
        self.dropout = nn.Dropout(dropout_rate if classifier_dropout is None else classifier_dropout)
        if gan_training:
            self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels + 1)
        else:
            self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()
        self.loss_fct = nn.BCEWithLogitsLoss()
        self.epsilon = epsilon
        self.gan_training = gan_training
        if self.gan_training:
            print("Training with GAN mode on!")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_mask: Optional[torch.Tensor] = None,
        external_states: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        labeled_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> ClassifierOutput:
        if input_ids is None and external_states is None:
            raise AssertionError("Empty input: input_ids and external states are empty")

        if input_ids is None:
            sequence_output = external_states
        else:
            outputs = self.encoder(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
            sequence_output = outputs.last_hidden_state[:, 0]  # get CLS embedding
            # add generator input to hidden states
            if external_states is not None:
                sequence_output = torch.cat([sequence_output, external_states], dim=0)

        sequence_output_drop = self.dropout(sequence_output)
        logits = self.classifier(sequence_output_drop)
        fake_probs, fake_logits = None, None
        if self.gan_training:
            fake_logits = logits[:, [-1]]
            fake_probs = self.sigmoid(fake_logits)
            logits = logits[:, :-1]
        probs = self.sigmoid(logits)
        loss = self.compute_loss(
            logits=logits, probs=probs, fake_probs=fake_probs, labels=labels, labeled_mask=labeled_mask
        )

        return NewClassifierOutput(
            loss=loss["real_loss"],
            fake_loss=loss["fake_loss"],
            logits=logits,
            fake_logits=fake_logits,
            probs=probs,
            fake_probs=fake_probs,
            hidden_states=sequence_output,
        )

    def compute_loss(
        self,
        logits: torch.Tensor,
        fake_logits: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        fake_probs: Optional[torch.Tensor] = None,
        labeled_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.FloatTensor]:
        real_loss = torch.FloatTensor([0]).cuda()
        fake_loss = torch.FloatTensor([0]).cuda()
        if labels is not None:
            if labeled_mask is not None:
                labeled_mask = labeled_mask.bool()
                logits = logits[labeled_mask]
                labels = labels[labeled_mask]
            if logits.shape[0] > 0:
                real_loss = self.loss_fct(logits, labels.float())
        if self.gan_training:
            fake_loss = -torch.mean(torch.log(fake_probs + self.epsilon))
        return {"real_loss": real_loss, "fake_loss": fake_loss}


class DiscriminatorForMultipleChoice(Discriminator):
    """Discriminator model for sequence classification tasks with transformer backbone"""

    def __init__(
        self,
        encoder_name: str,
        num_labels: int = 4,
        dropout_rate: Optional[float] = 0.15,
        ce_ignore_index: Optional[int] = -100,
        epsilon: Optional[float] = 1e-8,
        gan_training: bool = False,
        **kwargs,
    ):
        super(DiscriminatorForMultipleChoice, self).__init__()
        self.num_labels = num_labels
        self.encoder_name = encoder_name
        self.encoder = AutoModel.from_pretrained(encoder_name)
        classifier_dropout = (
            self.encoder.config.classifier_dropout
            if hasattr(self.encoder.config, "classifier_dropout")
            else None
        )
        self.dropout = nn.Dropout(dropout_rate if classifier_dropout is None else classifier_dropout)
        if gan_training:
            out_size_clf = 2
        else:
            out_size_clf = 1
        self.classifier = nn.Linear(self.encoder.config.hidden_size, out_size_clf)
        self.softmax = nn.Softmax(dim=-1)
        self.loss_fct = CrossEntropyLoss(ignore_index=ce_ignore_index)
        self.epsilon = epsilon
        self.gan_training = gan_training
        if self.gan_training:
            print("Training with GAN mode on!")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_mask: Optional[torch.Tensor] = None,
        external_states: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        labeled_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> ClassifierOutput:
        if input_ids is None and external_states is None:
            raise AssertionError("Empty input: input_ids and external states are empty")

        if input_ids is None:
            sequence_output = external_states
        else:
            input_ids = input_ids.view(-1, input_ids.size(-1))
            input_mask = input_mask.view(-1, input_mask.size(-1))
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            outputs = self.encoder(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
            sequence_output = outputs[1]  # pooled output
            # add generator input to hidden states
            if external_states is not None:
                sequence_output = torch.cat([sequence_output, external_states], dim=0)

        sequence_output_drop = self.dropout(sequence_output)
        logits = self.classifier(sequence_output_drop)
        fake_probs, fake_logits = None, None
        if self.gan_training:
            fake_probs = self.softmax(logits)[:, [1]]
            fake_logits = logits[:, [1]]
            logits = logits[:, [0]]
        if input_ids is not None:
            logits = logits.view(-1, self.num_labels)
        probs = self.softmax(logits)
        loss = self.compute_loss(
            logits=logits, probs=probs, fake_probs=fake_probs, labels=labels, labeled_mask=labeled_mask
        )

        return NewClassifierOutput(
            loss=loss["real_loss"],
            fake_loss=loss["fake_loss"],
            logits=logits,
            fake_logits=fake_logits,
            probs=probs,
            fake_probs=fake_probs,
            hidden_states=sequence_output,
        )

    def compute_loss(
        self,
        logits: torch.Tensor,
        fake_logits: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        fake_probs: Optional[torch.Tensor] = None,
        labeled_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.FloatTensor]:
        real_loss = torch.FloatTensor([0]).cuda()
        fake_loss = torch.FloatTensor([0]).cuda()
        if labels is not None:
            if labeled_mask is not None:
                labeled_mask = labeled_mask.bool()
                logits = logits[labeled_mask]
                labels = labels[labeled_mask]
            if logits.shape[0] > 0:
                real_loss = self.loss_fct(logits, labels)
        if self.gan_training:
            fake_loss = -torch.mean(torch.log(fake_probs + self.epsilon))
        return {"real_loss": real_loss, "fake_loss": fake_loss}


class DiscriminatorForTokenClassification(Discriminator):
    """Discriminator model for token classification tasks with transformer backbone"""

    def __init__(
        self,
        encoder_name: str,
        num_labels: int = 10,
        dropout_rate: Optional[float] = 0.15,
        ce_ignore_index: Optional[int] = -100,
        epsilon: Optional[float] = 1e-8,
        gan_training: bool = False,
        **kwargs,
    ):
        super(DiscriminatorForTokenClassification, self).__init__()
        self.num_labels = num_labels
        self.encoder_name = encoder_name
        self.encoder = AutoModel.from_pretrained(encoder_name)
        classifier_dropout = (
            self.encoder.config.classifier_dropout
            if hasattr(self.encoder.config, "classifier_dropout")
            else None
        )
        self.dropout = nn.Dropout(dropout_rate if classifier_dropout is None else classifier_dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.softmax = nn.Softmax(dim=-1)
        self.ignore_index = ce_ignore_index
        self.loss_fct = CrossEntropyLoss(ignore_index=self.ignore_index)
        self.epsilon = epsilon
        self.gan_training = gan_training
        if self.gan_training:
            print("Training with GAN mode on!")
            self.real_labels = torch.arange(num_labels) != (num_labels - 1)
            self.fake_index = -1
            print(f"Default fake label index is {self.fake_index}")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_mask: Optional[torch.Tensor] = None,
        external_states: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        labeled_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> ClassifierOutput:
        # simple check
        if input_ids is None and external_states is None:
            raise AssertionError("Empty input: input_ids and external states are empty")

        if input_ids is not None:
            outputs = self.encoder(input_ids, attention_mask=input_mask)
            sequence_output = outputs[0]

            # add generator input to hidden states
            if external_states is not None:
                sequence_output = torch.cat([sequence_output, external_states], dim=0)
        else:
            sequence_output = external_states

        sequence_output_drop = self.dropout(sequence_output)
        logits = self.classifier(sequence_output_drop)
        probs = self.softmax(logits)

        loss = self.compute_loss(logits=logits, probs=probs, labels=labels, labeled_mask=labeled_mask)

        return ClassifierOutput(loss=loss, logits=logits, probs=probs, hidden_states=sequence_output)

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        labeled_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.FloatTensor]:
        loss = None
        if labels is not None:
            if labeled_mask is not None:
                labeled_mask = labeled_mask.bool()
                logits = logits[labeled_mask]
                labels = labels[labeled_mask]
                if logits.shape[0] == 0:
                    return torch.FloatTensor([0]).cuda()

            if self.gan_training:
                logits = logits[:, :, self.real_labels]
                loss = self.loss_fct(logits.view(-1, self.num_labels - 1), labels.view(-1))
            else:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        elif self.gan_training:
            loss = -torch.mean(torch.log(probs[:, self.fake_index] + self.epsilon))
        return loss
