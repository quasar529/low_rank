import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
import os
import random
import sys
import transformers

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertTokenizer,
    BertForSequenceClassification,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import wandb
import copy
import torch.nn as nn
import math
from tqdm import tqdm, trange
from sklearn.metrics import f1_score
from transformers import EarlyStoppingCallback

# import loralib as lora
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    prepare_model_for_kbit_training,
)
from collections import Counter
import glob
import time
import datasets
from datasets import load_metric

MODEL_SAVE_REPO = "MayIBorn/qlora-llama-7b"
HUGGINGFACE_AUTH_TOKEN = "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"
os.environ["WANDB_PROJECT"] = "qlora"


def compute_metrics(eval_pred):
    metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def train():
    dataset_name = "sst2"
    """
    가능한 데이터 셋
    cola (The Corpus of Linguistic Acceptability)
    sst2 (The Stanford Sentiment Treebank)
    mrpc (The Microsoft Research Paraphrase Corpus)
    qqp (The Quora Question Pairs)
    stsb (The Semantic Textual Similarity Benchmark)
    mnli (The Multi-Genre Natural Language Inference)
    qnli (The Question Natural Language Inference)
    rte (The Recognizing Textual Entailment)
    wnli (The Winograd Schema Challenge)
    """
    dataset = datasets.load_dataset("glue", f"{dataset_name}")
    model_id = "huggyllama/llama-7b"
    tokenizer = AutoTokenizer.from_pretrained(f"{model_id}")

    output_dir = f"./qlora_{model_id}"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if dataset_name == "sst2":
        preprocess_function = lambda examples: tokenizer(examples["sentence"], truncation=True, padding=True)
    elif dataset_name == "rte":
        preprocess_function = lambda examples: tokenizer(
            examples["sentence1"], examples["sentence2"], truncation=True, padding=True
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    encoded_dataset = dataset.map(preprocess_function, batched=True)
    train_dataset = encoded_dataset["train"]
    val_dataset = encoded_dataset["validation"]
    test_dataset = encoded_dataset["test"]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, quantization_config=bnb_config, device_map={"": 0}, num_labels=2
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(r=8, lora_alpha=8, target_modules=["q_proj", "k_proj"], task_type="SEQ_CLS")
    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    datetime = time.strftime("%Y_%m_%d_%H_%M", time.localtime(time.time()))

    training_args = TrainingArguments(
        output_dir=f"{output_dir}",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=32,
        warmup_steps=500,
        num_train_epochs=1,
        learning_rate=4e-4,
        weight_decay=0.01,
        fp16=True,
        fp16_opt_level="O2",
        fp16_full_eval=True,
        do_eval=True,
        do_train=True,
        seed=42,
        save_total_limit=2,
        report_to="wandb",
        run_name=f"qlora_{datetime}_{model_id}",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=10,
        push_to_hub=True,
        label_smoothing_factor=0.1,
        evaluation_strategy="steps",
        eval_steps=400,
        save_steps=1200,
    )
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        compute_metrics=compute_metrics,
        # data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()
    model.push_to_hub(MODEL_SAVE_REPO, use_temp_dir=True, use_auth_token=HUGGINGFACE_AUTH_TOKEN)
    tokenizer.push_to_hub(MODEL_SAVE_REPO, use_temp_dir=True, use_auth_token=HUGGINGFACE_AUTH_TOKEN)

    trainer.evaluate()


def main():
    # eval_predict()
    train()


if __name__ == "__main__":
    main()
