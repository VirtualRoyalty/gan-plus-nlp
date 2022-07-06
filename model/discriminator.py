import base
from base import *
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers import AutoModel
from collections import OrderedDict


class DiscriminatorForTokenClassification(BaseModel):
    """Discriminator model class with transformer backbone"""

    def __init__(self,
                 encoder_name: str,
                 num_labels: int = 10,
                 dropout_rate: Optional[float] = 0.1,
                 **kwargs):
        super(DiscriminatorForTokenClassification, self).__init__()

        self.encoder = AutoModel.from_pretrained(encoder_name)
        classifier_dropout = (
            self.encoder.classifier_dropout if self.encoder.config.classifier_dropout is not None
            else dropout_rate
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                input_mask: Optional[torch.Tensor] = None,
                external_states: Optional[torch.Tensor] = None,
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
        return TokenClassifierOutput(logits=logits,
                                     probs=probs,
                                     hidden_states=sequence_output)


@dataclass
class TokenClassifierOutput(OrderedDict):
    logits: torch.FloatTensor = None
    probs: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None