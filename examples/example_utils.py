import numpy as np
from datasets import Dataset
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


def prepare_dataset(dataset, tokenizer, config, text_col="text"):
    tokenize = lambda x: tokenizer(x[text_col], truncation=True, max_length=config["max_seq_length"])
    tokenized_dataset = dataset.map(tokenize, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns([text_col])

    train_df = tokenized_dataset["train"].to_pandas()
    multiplier = int(np.log2(len(train_df) / train_df.labeled_mask.sum()))
    for _ in range(multiplier - 1):
        train_df = train_df.append(train_df[train_df.labeled_mask == True])
    tokenized_dataset["train"] = Dataset.from_pandas(train_df, preserve_index=False)
    return tokenized_dataset


def prepare_dataloaders(dataset, tokenizer, config, text_col="text"):
    tokenized_dataset = prepare_dataset(dataset, tokenizer, config, text_col)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        tokenized_dataset["train"],
        batch_size=config["batch_size"],
        sampler=RandomSampler(tokenized_dataset["train"]),
        collate_fn=data_collator,
        pin_memory=True,
    )

    valid_dataloader = DataLoader(
        tokenized_dataset["validation"],
        batch_size=config["batch_size"],
        sampler=SequentialSampler(tokenized_dataset["validation"]),
        collate_fn=data_collator,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        tokenized_dataset["test"],
        batch_size=config["batch_size"],
        sampler=SequentialSampler(tokenized_dataset["test"]),
        collate_fn=data_collator,
        pin_memory=True,
    )
    config["num_train_examples"] = len(train_dataloader.dataset)
    return {"train": train_dataloader, "test": test_dataloader, "valid": valid_dataloader}
