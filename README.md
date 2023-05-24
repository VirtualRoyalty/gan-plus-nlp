![tests](https://github.com/VirtualRoyalty/gan-token-classification/actions/workflows/ci.yml/badge.svg)

# 🦍 Semi-supervised learning for NLP via `GAN`
---

Semi-supervised learning for NLP tasks via GAN. Such approach can be used to enhance models in terms of small bunch of labeled examples.

### Supported tasks are following:

-  ✅ text classification (see `DiscriminatorForSequenceClassification`)
   - `+` multiple input text classification (e.g. NLI, paraprhase detection)
-  ✅ multi-label text classification (see `DiscriminatorForMultiLabelClassification`)
-  ✅ token classification (e.g. NER, see  `DiscriminatorForTokenClassification`)
-  ✅ multiple choice tasks (see `DiscriminatorForMultipleChoice`)


### Repo structure:
```
.
├── base
│   ├── __init__.py
│   ├── base_model.py
│   └── base_trainer.py
├── model
│   ├── __init__.py
│   ├── discriminator.py
│   ├── generator.py
│   └── utils.py
├── trainer
│   ├── __init__.py
│   ├── gan_trainer.py
│   └── trainer.py
├── tests
│   ├── __init__.py
│   ├── base_tests.py
│   ├── model_tests.py
│   └── trainer_tests.py
├── LICENSE
├── README.md
└── requirements.txt
```

---

```BibTex
@article{VirtualRoyalty,
  title   = "Semi-supervised learning for natural language processing via GAN.",
  author  = "Alperovich, Vadim",
  year    = "2023",
  url     = "https://github.com/VirtualRoyalty/gan-plus-nlp",
}
```
