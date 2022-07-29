from base import *
from dataclasses import dataclass
from typing import Optional, Tuple
from collections import OrderedDict

from torch.nn import CrossEntropyLoss
from transformers import AutoModel, AutoTokenizer, AutoConfig


class DiscriminatorForTokenClassification(BaseModel):
    """Discriminator model class with transformer backbone"""

    def __init__(self,
                 encoder_name: str,
                 num_labels: int = 10,
                 dropout_rate: Optional[float] = 0.15,
                 ignore_index: Optional[int] = -100,
                 **kwargs):
        super(DiscriminatorForTokenClassification, self).__init__()
        self.num_labels = num_labels
        self.encoder_name = encoder_name
        self.encoder = AutoModel.from_pretrained(encoder_name)
        classifier_dropout = (
            self.encoder.config.classifier_dropout
            if hasattr(self.encoder.config, "classifier_dropout")
            else None
        )
        self.dropout = nn.Dropout(dropout_rate
                                  if classifier_dropout is None
                                  else classifier_dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.softmax = nn.Softmax(dim=-1)
        self.ignore_index = ignore_index
        self.loss_fct = CrossEntropyLoss(ignore_index=self.ignore_index)

    def get_tokenizer(self) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(self.encoder_name)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                input_mask: Optional[torch.Tensor] = None,
                external_states: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                **kwargs):

        # simple check
        if input_ids is None and external_states is None:
            raise AssertionError('Empty input: input_ids and external states are empty')

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

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(loss=loss,
                                     logits=logits,
                                     probs=probs,
                                     hidden_states=sequence_output)

    def freeze_backbone(self) -> None:
        for name, parameter in self.encoder.named_parameters():
            parameter.requires_grad = False


@dataclass()
class TokenClassifierOutput(OrderedDict):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    probs: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None

    def __repr__(self):
        kws = [f"{key}={type(value).__name__}"
               for key, value in self.__dict__.items()]
        return f"{type(self).__name__}({', '.join(kws)})"

    def __str__(self):
        return self.__repr__()
