#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import argparse
from tokenizers import AddedToken
import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
import numpy as np
from datasets import DatasetDict
from os import listdir
from os.path import isfile, join

from utils_eval_metrics import compute_metrics_with_extra
from utils_loading_lucid import load_lucid


def get_args():
    parser = argparse.ArgumentParser(description="Training model parameters")

    # Arguments for modelling different scenarios
    parser.add_argument("--model_type", type=str, default="T5-small", help="Model to be used")
    parser.add_argument("--random_seed", type=int, default=42, help="Choose the random seed")

    # Reducing the dataset
    parser.add_argument("--reduce", type=int, default=0, help="Reduce the dataset or not")
    parser.add_argument(
        "--reduce_number",
        type=int,
        default=100,
        help="Number of training examples to train with if reducing",
    )

    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=8, help="Training batch size"
    )
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Eval batch size")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Epochs")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Schedule type")
    parser.add_argument("--learning_rate", type=float, default=5e-05, help="Learning rate")
    parser.add_argument(
        "--gradient_accumulation_steps", type=float, default=1, help="Gradient accumulation"
    )

    # Retrieval settings
    parser.add_argument(
        "--include_tools", type=int, default=0, help="If tools should be included in the prompt"
    )
    parser.add_argument("--oracle", type=int, default=1, help="If we have oracle tool retrieval")

    params, _ = parser.parse_known_args()

    return params


def find_all_intents_and_query_intents():
    # We find all our intents (minus held out intents)
    mypath = "../lucid_v1.0/toolbox_intents"
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    # We also find our heldout intents
    mypath = "../lucid_v1.0/toolbox_intents_heldout"
    files_heldout = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    # We outout a list of the intent names
    all_files = files + files_heldout
    all_intents = [x.replace(".json", "") for x in all_files]

    return all_intents


def set_seed(seed_value: int) -> None:
    """
    Set seed for reproducibility.

    Args:
        seed_value: chosen random seed
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def create_features(examples):
    enc_input = tokenizer.batch_encode_plus(examples["context"], truncation=True)
    enc_label = tokenizer.batch_encode_plus(examples["target"], truncation=True)

    input_ids = enc_input["input_ids"]
    attention_mask = enc_input["attention_mask"]
    label = enc_label["input_ids"]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label}


def model_init():
    return AutoModelForSeq2SeqLM.from_pretrained(params.model_type)


if __name__ == "__main__":
    params = get_args()
    params.reduce = bool(params.reduce)
    params.include_tools = bool(params.include_tools)
    params.oracle = bool(params.oracle)

    all_intents = find_all_intents_and_query_intents()

    # Load our dataset
    tokenizer = AutoTokenizer.from_pretrained(params.model_type, truncation_side="left")
    tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken("\n")]})
    data = load_lucid(tokenizer, all_intents, params.include_tools, params.oracle)

    # Processing dataset
    data = data.map(create_features, batched=True)
    data = data.remove_columns(["context", "target"])
    data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_data = data.filter(lambda x: x["split"] == "train")

    val_data = DatasetDict(
        {
            "dev": data.filter(lambda ex: ex["split"] == "dev"),
            "test": data.filter(lambda ex: ex["split"] == "test"),
            "test_ood": data.filter(lambda ex: ex["split"] == "test_ood"),
        }
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer)

    compute_metrics = compute_metrics_with_extra(tokenizer, all_intents)

    # Reducing and shuffling
    if params.reduce:
        train_data = train_data.shuffle(seed=params.random_seed).select(range(params.reduce_number))
    else:
        train_data = train_data.shuffle(seed=params.random_seed)

    # Creating model and trainer
    training_args = TrainingArguments(
        output_dir="./results",
        seed=params.random_seed,
        evaluation_strategy="epoch",
        include_inputs_for_metrics=True,
        per_device_train_batch_size=params.per_device_train_batch_size,
        per_device_eval_batch_size=params.per_device_eval_batch_size,
        num_train_epochs=params.num_train_epochs,
        learning_rate=params.learning_rate,
        lr_scheduler_type=params.lr_scheduler_type,
        gradient_accumulation_steps=params.gradient_accumulation_steps,
        eval_accumulation_steps=1,
    )

    trainer = Trainer(
        model_init=model_init,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics,
    )

    print("Training beginning...")

    trainer.evaluate(eval_dataset=val_data["test"])
    trainer.train()

    print("Training finished")

    trainer.save_model("/saved_llm/saved_llm")
