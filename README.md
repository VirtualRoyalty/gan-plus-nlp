![tests](https://github.com/VirtualRoyalty/gan-token-classification/actions/workflows/ci.yml/badge.svg)

# 🦍 Semi-supervised learning for NLP via `GAN`
---

Semi-supervised learning for NLP tasks via GAN. Such approach can be used to enhance models in terms of small bunch of labeled examples.

### Example usage

see detailed in [examples](https://github.com/VirtualRoyalty/gan-plus-nlp/blob/main/examples/sequence-classification.ipynb)

```python

import model
from trainer import gan_trainer as gan_trainer_module

...

config["encoder_name"] = "distilbert-base-uncased"

discriminator = model.DiscriminatorForSequenceClassification(**config)

generator = model.SimpleSequenceGenerator(
    input_size=config["noise_size"],
    output_size=discriminator.encoder.config.hidden_size,
)

gan_trainer = gan_trainer_module.GANTrainerSequenceClassification(
    config=config,
    discriminator=discriminator,
    generator=generator,
    train_dataloader=loaders["train"],
    valid_dataloader=loaders["valid"],
    device=device,
    save_path=config["save_path"],
)

for epoch_i in range(config["num_train_epochs"]):
    print(
        f"======== Epoch {epoch_i + 1} / {config['num_train_epochs']} ========"
    )
    train_info = gan_trainer.train_epoch(log_env=None)
    result = gan_trainer.validation(log_env=None)

...

predict_info = gan_trainer.predict(
    discriminator, loaders["test"], label_names=config["label_names"]
)
print(predict_info["overall_f1"])
```

### Supported tasks are following:

-  ✅ text classification (see `DiscriminatorForSequenceClassification`)
   - `+` multiple input text classification (e.g. NLI, paraprhase detection)
-  ✅ multi-label text classification (see `DiscriminatorForMultiLabelClassification`)
-  ✅ token classification (e.g. NER, see  `DiscriminatorForTokenClassification`)
-  ✅ multiple choice tasks (see `DiscriminatorForMultipleChoice`)


### Repo structure:
```
.
├── base/
│   ├── __init__.py
│   ├── base_model.py
│   └── base_trainer.py
├── model/
│   ├── __init__.py
│   ├── discriminator.py
│   ├── generator.py
│   └── utils.py
├── trainer/
│   ├── __init__.py
│   ├── gan_trainer.py
│   └── trainer.py
├── tests/
│   ├── __init__.py
│   ├── base_tests.py
│   ├── model_tests.py
│   └── trainer_tests.py
├── examples/
├── LICENSE
├── README.md
└── requirements.txt
```

This work based on [GAN-BERT: Generative Adversarial Learning for
Robust Text Classification with a Bunch of Labeled Examples](https://aclanthology.org/2020.acl-main.191.pdf), (Croce et al, 2020)

---

```BibTex
@article{VirtualRoyalty,
  title   = "Semi-supervised learning for natural language processing via GAN.",
  author  = "Alperovich, Vadim",
  year    = "2023",
  url     = "https://github.com/VirtualRoyalty/gan-plus-nlp",
}
```
