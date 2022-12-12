import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModel, AutoTokenizer, AutoConfig

from typing import Optional, Tuple

from base import BaseModel
from model.utils import TokenClassifierOutput


class DiscriminatorForTokenClassification(BaseModel):
    """Discriminator model class with transformer backbone"""

    def __init__(
        self,
        encoder_name: str,
        num_labels: int = 10,
        dropout_rate: Optional[float] = 0.15,
        ce_ignore_index: Optional[int] = -100,
        epsilon: Optional[float] = 1e-8,
        fake_label_index: Optional[int] = None,
        **kwargs
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
        self.fake_label_index = fake_label_index
        if self.fake_label_index is not None:
            fake_index = fake_label_index if fake_label_index >= 0 else num_labels - fake_label_index
            self.real_labels = torch.arange(num_labels) != fake_index

    def get_tokenizer(self) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(self.encoder_name)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_mask: Optional[torch.Tensor] = None,
        external_states: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        labeled_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> TokenClassifierOutput:

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

        return TokenClassifierOutput(loss=loss, logits=logits, probs=probs, hidden_states=sequence_output)

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
                    return 0

            if self.fake_label_index is not None:
                logits = logits[:, :, self.real_labels]
            else:
                logits = logits

            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        elif self.fake_label_index is not None:
            loss = -torch.mean(torch.log(probs[:, self.fake_label_index] + self.epsilon))
        return loss

    def freeze_backbone(self) -> None:
        for name, parameter in self.encoder.named_parameters():
            parameter.requires_grad = False
