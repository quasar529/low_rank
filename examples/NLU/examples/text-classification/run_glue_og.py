#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.
import copy
import logging
import os
import random
import sys
import torch
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EarlyStoppingCallback,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
import torch.nn as nn
from peft import get_peft_model, LoraConfig
import wandb
from peft import PeftModel, PeftConfig

HUGGINGFACE_AUTH_TOKEN = "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"
os.environ["WANDB_PROJECT"] = "DeBERTaV2_STSB"
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.4.0")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


class MRPCTrainer(Trainer):
    """
    The MRPCTrainer object is a subclass of Trainer that is designed for the MRPC dataset.

    Methods:
        training_step: Overriding the training_step method of Trainer.
    """

    def training_step(self, model, inputs):
        """
        Overriding the training_step method of Trainer.
        Batch마다 Class의 Ratio를 계산하여 Logging한다.
        """
        labels = inputs.get("labels")
        if labels is not None:
            total = len(labels)
            same_meaning = labels.sum().item()
            different_meaning = total - same_meaning
            ratio = same_meaning / total
            print(f"Number of same_meaning: {same_meaning}, different_meaning: {different_meaning}, ratio: {ratio}")

            if not 0.2 <= ratio <= 0.8:
                print(f"Warning: The ratio at step {self.state.global_step} is out of the 20-80 range.")

            wandb.log(
                {
                    "same_meaning": same_meaning,
                    "different_meaning": different_meaning,
                    "ratio": ratio,
                },
                step=self.state.global_step,
            )

        return super().training_step(model, inputs)


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


def deberta_init_dW_with_svd(model, approx_rank: int):
    """
    Pretrained Weight인 W를 SVD를 통해 분해 (U @ S @ Vt)
    U[:, :approx_rank] @ sqrt(S[:approx_rank])를 LoRA A, sqrt(S[:approx_rank]) @ Vt를 LoRA B의 Weight로 초기화한다.

    Args:
        model: DeBERTaV2 XXL
        approx_rank (int): SVD를 통해 분해할 Rank이자 LoRA A, LoRA B의 Rank
    """
    q_proj_v_loraA_weights = []
    k_proj_v_loraA_weights = []
    q_proj_v_loraB_weights = []
    k_proj_v_loraB_weights = []

    len_of_layers = len(model.deberta.encoder.layer)

    with torch.no_grad():
        for i in range(len_of_layers):
            q_original_weight = model.deberta.encoder.layer[i].attention.self.query_proj.weight.data.T
            k_original_weight = model.deberta.encoder.layer[i].attention.self.value_proj.weight.data.T

            q_proj_u, q_proj_s, q_proj_v = torch.linalg.svd(q_original_weight)
            k_proj_u, k_proj_s, k_proj_v = torch.linalg.svd(k_original_weight)

            q_proj_v_loraA_weights.append(q_proj_u[:, :approx_rank] @ torch.diag(q_proj_s[:approx_rank]).sqrt())

            k_proj_v_loraA_weights.append(k_proj_u[:, :approx_rank] @ torch.diag(k_proj_s[:approx_rank]).sqrt())

            q_proj_v_loraB_weights.append(torch.diag(q_proj_s[:approx_rank]).sqrt() @ q_proj_v[:approx_rank, :])

            k_proj_v_loraB_weights.append(torch.diag(k_proj_s[:approx_rank]).sqrt() @ k_proj_v[:approx_rank, :])

    with torch.no_grad():
        for i in range(len_of_layers):
            model.deberta.encoder.layer[i].attention.self.query_proj.lora_A.default.weight.data = copy.deepcopy(
                q_proj_v_loraA_weights[i].transpose(0, 1).contiguous()
            )
            model.deberta.encoder.layer[i].attention.self.value_proj.lora_A.default.weight.data = copy.deepcopy(
                k_proj_v_loraA_weights[i].transpose(0, 1).contiguous()
            )
            model.deberta.encoder.layer[i].attention.self.query_proj.lora_B.default.weight.data = copy.deepcopy(
                q_proj_v_loraB_weights[i].transpose(0, 1).contiguous()
            )

            model.deberta.encoder.layer[i].attention.self.value_proj.lora_B.default.weight.data = copy.deepcopy(
                k_proj_v_loraB_weights[i].transpose(0, 1).contiguous()
            )


def deberta_init_dW_with_svd_scaling(model, approx_rank: int):
    """
    Pretrained Weight인 W를 SVD를 통해 분해 (U @ S @ Vt)
    U[:, :approx_rank] @ sqrt(S)를 LoRA A, sqrt(S) @ Vt를 LoRA B의 Weight로 초기화한다.
    이 때, SVD 결과가 Gaussian과 비교했을 때 매우 크기 때문에
    선언 시 Gaussian으로 초기화 된 LoRA A와 vector-wise 비교해서 Scaling을 한다.

    Args:
        model : DeBERTaV2 XXL
        approx_rank (int): SVD를 통해 분해할 Rank이자 LoRA A, LoRA B의 Rank
    """
    len_of_layers = len(model.deberta.encoder.layer)
    q_new_lora_A_list = []
    v_new_lora_A_list = []

    q_new_lora_B_list = []
    v_new_lora_B_list = []

    for layer_idx in range(len_of_layers):
        q_original_weight = model.deberta.encoder.layer[layer_idx].attention.self.query_proj.weight.data.T
        v_original_weight = model.deberta.encoder.layer[layer_idx].attention.self.value_proj.weight.data.T

        q_og_lora_A = model.base_model.model.deberta.encoder.layer[
            layer_idx
        ].attention.self.query_proj.lora_A.default.weight.T
        v_og_lora_A = model.base_model.model.deberta.encoder.layer[
            layer_idx
        ].attention.self.value_proj.lora_A.default.weight.T

        q_proj_u, q_proj_s, q_proj_v = torch.linalg.svd(q_original_weight)
        q_new_lora_A = q_proj_u[:, :approx_rank] @ torch.diag(q_proj_s[:approx_rank]).sqrt()
        q_new_lora_B = torch.diag(q_proj_s[:approx_rank]).sqrt() @ q_proj_v[:approx_rank, :]

        v_proj_u, v_proj_s, v_proj_v = torch.linalg.svd(v_original_weight)
        v_new_lora_A = v_proj_u[:, :approx_rank] @ torch.diag(v_proj_s[:approx_rank]).sqrt()
        v_new_lora_B = torch.diag(v_proj_s[:approx_rank]).sqrt() @ v_proj_v[:approx_rank, :]

        for i in range(4):
            print(f"Before Scale, q {i}th col Norm", torch.norm(q_new_lora_A[:, i]))

            q_og_lora_A_icol_norm = torch.norm(q_og_lora_A[:, i])
            q_new_lora_A_icol_norm = torch.norm(q_new_lora_A[:, i])

            q_scale = q_og_lora_A_icol_norm / q_new_lora_A_icol_norm

            q_new_lora_A[:, i] = q_new_lora_A[:, i] * q_scale
            q_new_lora_B[:, i] = q_new_lora_B[:, i] * q_scale

            print(f"After Scale, q {i}th col Norm", torch.norm(q_new_lora_A[:, i]))
            print("-" * 50)

            print(f"Before Scale, v {i}th col Norm", torch.norm(v_new_lora_A[:, i]))

            v_og_lora_A_icol_norm = torch.norm(v_og_lora_A[:, i])
            v_new_lora_A_icol_norm = torch.norm(v_new_lora_A[:, i])

            v_scale = v_og_lora_A_icol_norm / v_new_lora_A_icol_norm

            v_new_lora_A[:, i] = v_new_lora_A[:, i] * v_scale
            v_new_lora_B[:, i] = v_new_lora_B[:, i] * v_scale

            print(f"After Scale, v {i}th col Norm", torch.norm(v_new_lora_A[:, i]))
            print("-" * 50)

        q_new_lora_A_list.append(q_new_lora_A)
        v_new_lora_A_list.append(v_new_lora_A)
        q_new_lora_B_list.append(q_new_lora_B)
        v_new_lora_B_list.append(v_new_lora_B)

        model.base_model.model.deberta.encoder.layer[
            layer_idx
        ].attention.self.query_proj.lora_A.default.weight.data = (
            q_new_lora_A_list[layer_idx].transpose(0, 1).contiguous()
        )
        model.base_model.model.deberta.encoder.layer[
            layer_idx
        ].attention.self.value_proj.lora_A.default.weight.data = (
            v_new_lora_A_list[layer_idx].transpose(0, 1).contiguous()
        )

        model.base_model.model.deberta.encoder.layer[
            layer_idx
        ].attention.self.query_proj.lora_B.default.weight.data = (
            q_new_lora_B_list[layer_idx].transpose(0, 1).contiguous()
        )
        model.base_model.model.deberta.encoder.layer[
            layer_idx
        ].attention.self.value_proj.lora_B.default.weight.data = (
            v_new_lora_B_list[layer_idx].transpose(0, 1).contiguous()
        )


def deberta_init_dW_A_with_svd(model, approx_rank: int):
    """
    Pretrained Weight인 W를 SVD를 통해 분해 (U @ S @ Vt)
    U[:, :approx_rank] @ sqrt(S[:approx_rank])를 LoRA A로 초기화한다.
    LoRA B는 0으로 초기화한다.

    Args:
        model : DeBERTaV2 XXL
        approx_rank (int): SVD를 통해 분해할 Rank이자 LoRA A, LoRA B의 Rank
    """
    q_proj_v_loraA_weights = []
    k_proj_v_loraA_weights = []

    len_of_layers = len(model.deberta.encoder.layer)
    with torch.no_grad():
        for i in range(len_of_layers):
            q_original_weight = model.deberta.encoder.layer[i].attention.self.query_proj.weight.data.T
            k_original_weight = model.deberta.encoder.layer[i].attention.self.value_proj.weight.data.T

            q_proj_u, q_proj_s, q_proj_v = torch.linalg.svd(q_original_weight)
            k_proj_u, k_proj_s, k_proj_v = torch.linalg.svd(k_original_weight)

            # w_q_encoder

            q_proj_v_loraA_weights.append(q_proj_u[:, :approx_rank] @ torch.diag(q_proj_s[:approx_rank]).sqrt())

            # w_v_encoder
            k_proj_v_loraA_weights.append(k_proj_u[:, :approx_rank] @ torch.diag(k_proj_s[:approx_rank]).sqrt())

    with torch.no_grad():
        for i in range(len_of_layers):
            model.deberta.encoder.layer[i].attention.self.query_proj.lora_A.default.weight.data = copy.deepcopy(
                q_proj_v_loraA_weights[i].transpose(0, 1).contiguous()
            )
            model.deberta.encoder.layer[i].attention.self.value_proj.lora_A.default.weight.data = copy.deepcopy(
                k_proj_v_loraA_weights[i].transpose(0, 1).contiguous()
            )


def deberta_init_dW_A_with_svd_scaling(model, approx_rank: int):
    """
    Pretrained Weight인 W를 SVD를 통해 분해 (U @ S @ Vt)
    U[:, :approx_rank] @ sqrt(S[:approx_rank])를 LoRA A로 초기화한다.
    LoRA B는 0으로 초기화한다.

    이 때, SVD 결과가 Gaussian과 비교했을 때 매우 크기 때문에
    선언 시 Gaussian으로 초기화 된 LoRA A와 vector-wise 비교해서 Scaling을 한다.

    Args:
        model : DeBERTaV2 XXL
        approx_rank (int): SVD를 통해 분해할 Rank이자 LoRA A, LoRA B의 Rank
    """
    len_of_layers = len(model.deberta.encoder.layer)
    q_new_lora_A_list = []
    v_new_lora_A_list = []

    for layer_idx in range(len_of_layers):
        q_original_weight = model.deberta.encoder.layer[layer_idx].attention.self.query_proj.weight.data.T
        v_original_weight = model.deberta.encoder.layer[layer_idx].attention.self.value_proj.weight.data.T

        q_og_lora_A = model.base_model.model.deberta.encoder.layer[
            layer_idx
        ].attention.self.query_proj.lora_A.default.weight.T
        v_og_lora_A = model.base_model.model.deberta.encoder.layer[
            layer_idx
        ].attention.self.value_proj.lora_A.default.weight.T

        q_proj_u, q_proj_s, q_proj_v = torch.linalg.svd(q_original_weight)
        q_new_lora_A = q_proj_u[:, :approx_rank] @ torch.diag(q_proj_s[:approx_rank]).sqrt()

        v_proj_u, v_proj_s, v_proj_v = torch.linalg.svd(v_original_weight)
        v_new_lora_A = v_proj_u[:, :approx_rank] @ torch.diag(v_proj_s[:approx_rank]).sqrt()

        for i in range(approx_rank):
            print(f"Before Scale, q {i}th col Norm", torch.norm(q_new_lora_A[:, i]))

            q_og_lora_A_icol_norm = torch.norm(q_og_lora_A[:, i])
            q_new_lora_A_icol_norm = torch.norm(q_new_lora_A[:, i])

            q_scale = q_og_lora_A_icol_norm / q_new_lora_A_icol_norm

            q_new_lora_A[:, i] = q_new_lora_A[:, i] * q_scale

            print(f"After Scale, q {i}th col Norm", torch.norm(q_new_lora_A[:, i]))
            print("####################")

            print(f"Before Scale, v {i}th col Norm", torch.norm(v_new_lora_A[:, i]))

            v_og_lora_A_icol_norm = torch.norm(v_og_lora_A[:, i])
            v_new_lora_A_icol_norm = torch.norm(v_new_lora_A[:, i])

            v_scale = v_og_lora_A_icol_norm / v_new_lora_A_icol_norm

            v_new_lora_A[:, i] = v_new_lora_A[:, i] * v_scale

            print(f"After Scale, v {i}th col Norm", torch.norm(v_new_lora_A[:, i]))
            print("####################")

        q_new_lora_A_list.append(q_new_lora_A)
        v_new_lora_A_list.append(v_new_lora_A)

        model.base_model.model.deberta.encoder.layer[
            layer_idx
        ].attention.self.query_proj.lora_A.default.weight.data = (
            q_new_lora_A_list[layer_idx].transpose(0, 1).contiguous()
        )
        model.base_model.model.deberta.encoder.layer[
            layer_idx
        ].attention.self.value_proj.lora_A.default.weight.data = (
            v_new_lora_A_list[layer_idx].transpose(0, 1).contiguous()
        )


def deberta_init_dW_with_svd_from_back(model, approx_rank: int):
    """
    Pretrained Weight인 W를 SVD를 통해 분해 (U @ S @ Vt)
    단 이 때, 성분을 앞에서 가져오지 않고 뒤에서 가져와 기존과 반대 방향의 성분을 얻는다
    U[:, -approx_rank:] @ sqrt(S[-approx_rank:])를 LoRA A, sqrt(S[-approx_rank:]) @ Vt[:, -approx_rank:]를 LoRA B의 Weight로 초기화한다.

    Args:
        model: DeBERTaV2 XXL
        approx_rank (int): SVD를 통해 분해할 Rank이자 LoRA A, LoRA B의 Rank

    """
    q_proj_v_loraB_weights = []
    k_proj_v_loraB_weights = []
    q_proj_v_loraA_weights = []
    k_proj_v_loraA_weights = []
    len_of_layers = len(model.deberta.encoder.layer)

    with torch.no_grad():
        for i in range(len_of_layers):
            q_original_weight = model.deberta.encoder.layer[i].attention.self.query_proj.weight.data.T
            k_original_weight = model.deberta.encoder.layer[i].attention.self.value_proj.weight.data.T

            q_proj_u, q_proj_s, q_proj_v = torch.linalg.svd(q_original_weight)
            k_proj_u, k_proj_s, k_proj_v = torch.linalg.svd(k_original_weight)

            q_proj_v_loraA_weights.append(q_proj_u[:, -approx_rank:] @ torch.diag(q_proj_s[-approx_rank:]).sqrt())
            q_proj_v_loraB_weights.append(torch.diag(q_proj_s[-approx_rank:]).sqrt() @ q_proj_v[-approx_rank:, :])

            k_proj_v_loraA_weights.append(k_proj_u[:, -approx_rank:] @ torch.diag(k_proj_s[-approx_rank:]).sqrt())
            k_proj_v_loraB_weights.append(torch.diag(k_proj_s[-approx_rank:]).sqrt() @ k_proj_v[-approx_rank:, :])

    with torch.no_grad():
        for i in range(len_of_layers):
            model.deberta.encoder.layer[i].attention.self.query_proj.lora_A.default.weight.data = copy.deepcopy(
                q_proj_v_loraA_weights[i].transpose(0, 1).contiguous()
            )
            model.deberta.encoder.layer[i].attention.self.value_proj.lora_A.default.weight.data = copy.deepcopy(
                k_proj_v_loraA_weights[i].transpose(0, 1).contiguous()
            )
            model.deberta.encoder.layer[i].attention.self.query_proj.lora_B.default.weight.data = copy.deepcopy(
                q_proj_v_loraB_weights[i].transpose(0, 1).contiguous()
            )

            model.deberta.encoder.layer[i].attention.self.value_proj.lora_B.default.weight.data = copy.deepcopy(
                k_proj_v_loraB_weights[i].transpose(0, 1).contiguous()
            )


def deberta_init_dW_A_with_svd_from_back(model, model_original, approx_rank: int):
    """
    Pretrained Weight인 W를 SVD를 통해 분해 (U @ S @ Vt)
    단 이 때, 성분을 앞에서 가져오지 않고 뒤에서 가져와 기존과 반대 방향의 성분을 얻는다
    U[:, -approx_rank:] @ sqrt(S[-approx_rank:])를 LoRA A로 초기화한다.
    LoRA B는 0으로 초기화한다.

    Args:
        model: DeBERTaV2 XXL
        approx_rank (int): SVD를 통해 분해할 Rank이자 LoRA A, LoRA B의 Rank

    """
    q_proj_v_loraA_weights = []
    k_proj_v_loraA_weights = []
    len_of_layers = len(model_original.deberta.encoder.layer)

    with torch.no_grad():
        for i in range(len_of_layers):
            q_original_weight = model_original.deberta.encoder.layer[i].attention.self.query_proj.weight.data.T
            k_original_weight = model_original.deberta.encoder.layer[i].attention.self.value_proj.weight.data.T

            q_proj_u, q_proj_s, q_proj_v = torch.linalg.svd(q_original_weight)
            k_proj_u, k_proj_s, k_proj_v = torch.linalg.svd(k_original_weight)

            q_proj_v_loraA_weights.append(q_proj_u[:, -approx_rank:] @ torch.diag(q_proj_s[-approx_rank:]).sqrt())

            k_proj_v_loraA_weights.append(k_proj_u[:, -approx_rank:] @ torch.diag(k_proj_s[-approx_rank:]).sqrt())

    with torch.no_grad():
        for i in range(len_of_layers):
            model.deberta.encoder.layer[i].attention.self.query_proj.lora_A.default.weight.data = copy.deepcopy(
                q_proj_v_loraA_weights[i].transpose(0, 1).contiguous()
            )
            model.deberta.encoder.layer[i].attention.self.value_proj.lora_A.default.weight.data = copy.deepcopy(
                k_proj_v_loraA_weights[i].transpose(0, 1).contiguous()
            )


def deberta_init_dW_B_with_svd_from_back(model, model_original, approx_rank: int):
    """
    Pretrained Weight인 W를 SVD를 통해 분해 (U @ S @ Vt)
    단 이 때, 성분을 앞에서 가져오지 않고 뒤에서 가져와 기존과 반대 방향의 성분을 얻는다
    sqrt(S[-approx_rank:]) @ Vt[:, -approx_rank:]를 LoRA B의 Weight로 초기화한다.
    LoRA A는 Gaussian으로 초기화한다.

    Args:
        model: DeBERTaV2 XXL
        approx_rank (int): SVD를 통해 분해할 Rank이자 LoRA A, LoRA B의 Rank

    """
    q_proj_v_loraB_weights = []
    k_proj_v_loraB_weights = []
    len_of_layers = len(model_original.deberta.encoder.layer)
    with torch.no_grad():
        for i in range(len_of_layers):
            q_original_weight = model_original.deberta.encoder.layer[i].attention.self.query_proj.weight.data.T
            k_original_weight = model_original.deberta.encoder.layer[i].attention.self.value_proj.weight.data.T

            q_proj_u, q_proj_s, q_proj_v = torch.linalg.svd(q_original_weight)
            k_proj_u, k_proj_s, k_proj_v = torch.linalg.svd(k_original_weight)

            q_proj_v_loraB_weights.append(torch.diag(q_proj_s[-approx_rank:]).sqrt() @ q_proj_v[-approx_rank:, :])

            k_proj_v_loraB_weights.append(torch.diag(k_proj_s[-approx_rank:]).sqrt() @ k_proj_v[-approx_rank:, :])

    with torch.no_grad():
        for i in range(len_of_layers):
            model.deberta.encoder.layer[i].attention.self.query_proj.lora_B.default.weight.data = copy.deepcopy(
                q_proj_v_loraB_weights[i].transpose(0, 1).contiguous()
            )

            model.deberta.encoder.layer[i].attention.self.value_proj.lora_B.default.weight.data = copy.deepcopy(
                k_proj_v_loraB_weights[i].transpose(0, 1).contiguous()
            )


def deberta_init_dW_B_with_svd_from_back_dW_A_zero(model, model_original, approx_rank: int):
    """
    Pretrained Weight인 W를 SVD를 통해 분해 (U @ S @ Vt)
    단 이 때, 성분을 앞에서 가져오지 않고 뒤에서 가져와 기존과 반대 방향의 성분을 얻는다
    sqrt(S[-approx_rank:]) @ Vt[:, -approx_rank:]를 LoRA B의 Weight로 초기화한다.
    LoRA A는 0으로 초기화한다.

    Args:
        model: DeBERTaV2 XXL
        approx_rank (int): SVD를 통해 분해할 Rank이자 LoRA A, LoRA B의 Rank

    """
    q_proj_v_loraB_weights = []
    k_proj_v_loraB_weights = []
    len_of_layers = len(model_original.deberta.encoder.layer)
    with torch.no_grad():
        for i in range(len_of_layers):
            q_original_weight = model_original.deberta.encoder.layer[i].attention.self.query_proj.weight.data.T
            k_original_weight = model_original.deberta.encoder.layer[i].attention.self.value_proj.weight.data.T

            q_proj_u, q_proj_s, q_proj_v = torch.linalg.svd(q_original_weight)
            k_proj_u, k_proj_s, k_proj_v = torch.linalg.svd(k_original_weight)

            q_proj_v_loraB_weights.append(torch.diag(q_proj_s[-approx_rank:]).sqrt() @ q_proj_v[-approx_rank:, :])

            k_proj_v_loraB_weights.append(torch.diag(k_proj_s[-approx_rank:]).sqrt() @ k_proj_v[-approx_rank:, :])

    with torch.no_grad():
        for i in range(len_of_layers):
            model.deberta.encoder.layer[i].attention.self.query_proj.lora_B.default.weight.data = copy.deepcopy(
                q_proj_v_loraB_weights[i].transpose(0, 1).contiguous()
            )

            model.deberta.encoder.layer[i].attention.self.value_proj.lora_B.default.weight.data = copy.deepcopy(
                k_proj_v_loraB_weights[i].transpose(0, 1).contiguous()
            )
            nn.init.zeros_(model.deberta.encoder.layer[i].attention.self.query_proj.lora_A.default.weight.data)
            nn.init.zeros_(model.deberta.encoder.layer[i].attention.self.value_proj.lora_A.default.weight.data)


def deberta_init_dW_A_with_svd_from_back_with_scaling(model, model_original, approx_rank: int):
    """
    Pretrained Weight인 W를 SVD를 통해 분해 (U @ S @ Vt)
    단 이 때, 성분을 앞에서 가져오지 않고 뒤에서 가져와 기존과 반대 방향의 성분을 얻는다
    U[:, -approx_rank:] @ sqrt(S[-approx_rank:])를 LoRA A로 초기화한다.

    이 때, SVD 결과의 뒤에서 가져온 성분이 Gaussian과 비교했을 때 매우 작기 때문에
    값을 키우는 Scaling을 한다.

    LoRA B는 0으로 초기화한다.

    Args:
        model: DeBERTaV2 XXL
        approx_rank (int): SVD를 통해 분해할 Rank이자 LoRA A, LoRA B의 Rank

    """
    q_proj_v_loraA_weights = []
    k_proj_v_loraA_weights = []
    len_of_layers = len(model_original.deberta.encoder.layer)

    with torch.no_grad():
        for i in range(len_of_layers):
            # 분자 : q_numerator, k_numerator
            # 분모 : q_denominator, k_denominator

            # Query
            q_numerator = 0
            q_denominator = 0

            # Key
            k_numerator = 0
            k_denominator = 0

            # 스케일링 : q_scaling, k_scaling
            q_scaling = 0
            k_scaling = 0

            q_original_weight = model_original.deberta.encoder.layer[i].attention.self.query_proj.weight.data.T
            k_original_weight = model_original.deberta.encoder.layer[i].attention.self.value_proj.weight.data.T

            # W = U * S * V.T
            q_proj_u, q_proj_s, q_proj_v = torch.linalg.svd(q_original_weight)
            k_proj_u, k_proj_s, k_proj_v = torch.linalg.svd(k_original_weight)

            # 연산을 위해 cpu로 옮김
            q_proj_s_cpu = q_proj_s.cpu().numpy()
            k_proj_s_cpu = k_proj_s.cpu().numpy()

            for i in range(approx_rank):
                # print(f"{i}번째 singular value: {q_proj_s_cpu[i]}")
                # print(f"{-(i+1)}번째 singular value: {q_proj_s_cpu[-(i+1)]}")

                # q_numerator = sigma_1^2 + sigma_2^2 + ... + sigma_r^2
                # q_denominator = sigma_(n-r+1)^2 + sigma_(n-r+2)^2 + ... + sigma_n^2
                q_numerator += np.power(q_proj_s_cpu[i], 2)
                q_denominator += np.power(q_proj_s_cpu[-(i + 1)], 2)

                # k_numerator = sigma_1^2 + sigma_2^2 + ... + sigma_r^2
                # k_denominator = sigma_(n-r+1)^2 + sigma_(n-r+2)^2 + ... + sigma_n^2
                k_numerator += np.power(k_proj_s_cpu[i], 2)
                k_denominator += np.power(k_proj_s_cpu[-(i + 1)], 2)

            q_scaling = np.sqrt(q_numerator / q_denominator)
            k_scaling = np.sqrt(k_numerator / k_denominator)

            # print(f"q_scaling: {q_scaling}")
            # print(f"k_scaling: {k_scaling}")

            # 기존 singular value에 스케일링
            q_proj_s *= q_scaling
            k_proj_s *= k_scaling

            q_proj_v_loraA_weights.append(q_proj_u[:, -approx_rank:] @ torch.diag(q_proj_s[-approx_rank:]).sqrt())
            k_proj_v_loraA_weights.append(k_proj_u[:, -approx_rank:] @ torch.diag(k_proj_s[-approx_rank:]).sqrt())

    with torch.no_grad():
        for i in range(len_of_layers):
            # lora_A의 weight를 초기화
            model.deberta.encoder.layer[i].attention.self.query_proj.lora_A.default.weight.data = copy.deepcopy(
                q_proj_v_loraA_weights[i].transpose(0, 1).contiguous()
            )
            model.deberta.encoder.layer[i].attention.self.value_proj.lora_A.default.weight.data = copy.deepcopy(
                k_proj_v_loraA_weights[i].transpose(0, 1).contiguous()
            )


def deberta_init_dW_A_with_svd_from_back_with_scaling_entire(model, model_original, approx_rank: int):
    q_proj_v_loraA_weights = []
    k_proj_v_loraA_weights = []
    len_of_layers = len(model_original.deberta.encoder.layer)

    with torch.no_grad():
        for i in range(len_of_layers):
            q_numerator = 0
            q_denominator = 0
            k_numerator = 0
            k_denominator = 0
            q_scaling = 0
            k_scaling = 0

            q_original_weight = model_original.deberta.encoder.layer[i].attention.self.query_proj.weight.data.T
            k_original_weight = model_original.deberta.encoder.layer[i].attention.self.value_proj.weight.data.T

            q_proj_u, q_proj_s, q_proj_v = torch.linalg.svd(q_original_weight)
            k_proj_u, k_proj_s, k_proj_v = torch.linalg.svd(k_original_weight)
            q_proj_s_cpu = q_proj_s.cpu().numpy()
            k_proj_s_cpu = k_proj_s.cpu().numpy()

            for i in range(len(q_proj_s)):
                q_numerator += np.power(q_proj_s_cpu[i], 2)
                k_numerator += np.power(k_proj_s_cpu[i], 2)

            for i in range(approx_rank):
                q_denominator += np.power(q_proj_s_cpu[-(i + 1)], 2)
                k_denominator += np.power(k_proj_s_cpu[-(i + 1)], 2)

            q_scaling = np.sqrt(q_numerator / q_denominator)
            k_scaling = np.sqrt(k_numerator / k_denominator)
            print(f"q_scaling: {q_scaling}")
            print(f"k_scaling: {k_scaling}")

            q_proj_s *= q_scaling
            k_proj_s *= k_scaling

            q_proj_v_loraA_weights.append(q_proj_u[:, -approx_rank:] @ torch.diag(q_proj_s[-approx_rank:]).sqrt())

            k_proj_v_loraA_weights.append(k_proj_u[:, -approx_rank:] @ torch.diag(k_proj_s[-approx_rank:]).sqrt())

    with torch.no_grad():
        for i in range(len_of_layers):
            model.deberta.encoder.layer[i].attention.self.query_proj.lora_A.default.weight.data = copy.deepcopy(
                q_proj_v_loraA_weights[i].transpose(0, 1).contiguous()
            )
            model.deberta.encoder.layer[i].attention.self.value_proj.lora_A.default.weight.data = copy.deepcopy(
                k_proj_v_loraA_weights[i].transpose(0, 1).contiguous()
            )


def deberta_init_dW_B_with_svd_by_head(model, approx_rank: int):
    len_of_layers = len(model.base_model.model.deberta.encoder.layer)

    for i in range(len_of_layers):
        q_loraB_head_list = []
        v_loraB_head_list = []

        q_weight = model.base_model.model.deberta.encoder.layer[i].attention.self.query_proj.weight.T
        v_weight = model.base_model.model.deberta.encoder.layer[i].attention.self.value_proj.weight.T

        all_q_head = copy.deepcopy(q_weight)
        # all_q_head = all_q_head.view(24, 1536, 64)
        all_q_head = all_q_head.reshape(24, 1536, 64)
        all_v_head = copy.deepcopy(v_weight)
        # all_v_head = all_v_head.view(24, 1536, 64)
        all_v_head = all_v_head.reshape(24, 1536, 64)
        for j in range(24):
            q_head = all_q_head[j]
            q_u, q_s, q_vt = torch.linalg.svd(q_head)
            q_loraB_head = torch.diag(q_s[:approx_rank]) @ q_vt[:approx_rank, :]

            v_head = all_v_head[j]
            v_u, v_s, v_vt = torch.linalg.svd(v_head)
            v_loraB_head = torch.diag(v_s[:approx_rank]) @ v_vt[:approx_rank, :]

            q_loraB_head_list.append(q_loraB_head)
            v_loraB_head_list.append(v_loraB_head)
        print(
            f"Complete Merge Head! len of q_loraB_head_list: {len(q_loraB_head_list)},len of v_loraB_head_list: {len(v_loraB_head_list)}"
        )
        q_loraB = torch.cat(q_loraB_head_list, dim=1)
        v_loraB = torch.cat(v_loraB_head_list, dim=1)

        # LoRA B
        model.base_model.model.deberta.encoder.layer[
            i
        ].attention.self.query_proj.lora_B.default.weight.data = q_loraB.T.contiguous()
        model.base_model.model.deberta.encoder.layer[
            i
        ].attention.self.value_proj.lora_B.default.weight.data = v_loraB.T.contiguous()

        # LoRA A
        model.base_model.model.deberta.encoder.layer[
            i
        ].attention.self.query_proj.lora_A.default.weight.data = torch.zeros_like(
            model.base_model.model.deberta.encoder.layer[i].attention.self.query_proj.lora_A.default.weight.data
        )
        model.base_model.model.deberta.encoder.layer[
            i
        ].attention.self.value_proj.lora_A.default.weight.data = torch.zeros_like(
            model.base_model.model.deberta.encoder.layer[i].attention.self.value_proj.lora_A.default.weight.data
        )

        print(f"Complete init LoraB! layer: {i}, q_loraB: {q_loraB.shape}, v_loraB: {v_loraB.shape}")


def deberta_init_dW_B_with_svd_us_by_head(model, approx_rank: int):
    len_of_layers = len(model.base_model.model.deberta.encoder.layer)
    for i in range(len_of_layers):
        q_loraB_head_list = []
        v_loraB_head_list = []

        q_weight = model.base_model.model.deberta.encoder.layer[i].attention.self.query_proj.weight.T
        v_weight = model.base_model.model.deberta.encoder.layer[i].attention.self.value_proj.weight.T

        all_q_head = copy.deepcopy(q_weight)
        all_q_head = all_q_head.reshape(approx_rank, 1536, 128)

        all_v_head = copy.deepcopy(v_weight)
        all_v_head = all_v_head.reshape(approx_rank, 1536, 128)

        for j in range(approx_rank):
            q_head = all_q_head[j]
            q_u, q_s, q_vt = torch.linalg.svd(q_head)
            q_loraB_head = q_u[:, :1] @ torch.diag(q_s[:1])

            v_head = all_v_head[j]
            v_u, v_s, v_vt = torch.linalg.svd(v_head)
            v_loraB_head = v_u[:, :1] @ torch.diag(v_s[:1])

            q_loraB_head_list.append(q_loraB_head)
            v_loraB_head_list.append(v_loraB_head)
        print(
            f"Complete Merge Head! len of q_loraB_head_list: {len(q_loraB_head_list)},len of v_loraB_head_list: {len(v_loraB_head_list)}"
        )
        q_loraB = torch.cat(q_loraB_head_list, dim=1)
        v_loraB = torch.cat(v_loraB_head_list, dim=1)

        model.base_model.model.deberta.encoder.layer[
            i
        ].attention.self.query_proj.lora_B.default.weight.data = q_loraB.contiguous()
        model.base_model.model.deberta.encoder.layer[
            i
        ].attention.self.value_proj.lora_B.default.weight.data = v_loraB.contiguous()

        model.base_model.model.deberta.encoder.layer[
            i
        ].attention.self.query_proj.lora_A.default.weight.data = torch.zeros_like(
            model.base_model.model.deberta.encoder.layer[i].attention.self.query_proj.lora_A.default.weight.data
        )
        model.base_model.model.deberta.encoder.layer[
            i
        ].attention.self.value_proj.lora_A.default.weight.data = torch.zeros_like(
            model.base_model.model.deberta.encoder.layer[i].attention.self.value_proj.lora_A.default.weight.data
        )

        print(f"Complete init LoraB! layer: {i}, q_loraB: {q_loraB.shape}, v_loraB: {v_loraB.shape}")


def deberta_init_dW_A_with_span(model, model_original, approx_rank: int):
    model.cpu()
    model_original.cpu()
    len_of_layers = len(model_original.deberta.encoder.layer)

    for i in range(len_of_layers):
        q_original_weight = model.base_model.model.deberta.encoder.layer[i].attention.self.query_proj.weight.data.T
        v_original_weight = model.base_model.model.deberta.encoder.layer[i].attention.self.value_proj.weight.data.T

        q_u, q_s, q_vt = torch.linalg.svd(q_original_weight)
        v_u, v_s, v_vt = torch.linalg.svd(v_original_weight)
        q_loraA_list = []
        v_loraA_list = []
        for forA in range(approx_rank):
            q_tilde_A = torch.zeros(1536)
            v_tilde_A = torch.zeros(1536)

            for j in range(forA, approx_rank):
                q_loraA_gaussain = copy.deepcopy(
                    model.base_model.model.deberta.encoder.layer[
                        i
                    ].attention.self.query_proj.lora_A.default.weight.data[j]
                )
                q_scailng = torch.dot(q_loraA_gaussain, q_u[j])
                component_q_tilde_a = q_scailng * q_u[j]

                v_loraA_gaussain = copy.deepcopy(
                    model.base_model.model.deberta.encoder.layer[
                        i
                    ].attention.self.value_proj.lora_A.default.weight.data[j]
                )
                v_scailng = torch.dot(v_loraA_gaussain, q_u[j])
                component_k_tilde_a = v_scailng * v_u[j]

                q_tilde_A += component_q_tilde_a
                v_tilde_A += component_k_tilde_a
            # q_normalized =  torch.norm() / torch.norm()
            q_loraA_list.append(q_tilde_A)
            v_loraA_list.append(v_tilde_A)
        q_loraA_tensor = torch.stack(q_loraA_list)
        v_loraA_tensor = torch.stack(v_loraA_list)

        model.base_model.model.deberta.encoder.layer[
            i
        ].attention.self.query_proj.lora_A.default.weight.data = q_loraA_tensor.contiguous()
        model.base_model.model.deberta.encoder.layer[
            i
        ].attention.self.value_proj.lora_A.default.weight.data = v_loraA_tensor.contiguous()


"""
Head-Wise SVD Initialization
"""


def deberta_init_dW_A_with_svd_us_by_head(model, approx_rank: int):
    """
    Pretrained W를 Head-Wise로 나눈 후
    각 Head를 SVD를 통해 분해 (U @ S @ Vt)해서 Head 마다 [Hidden_dim, 1]하나를 뽑아낸다.
    이를 Concat하여 LoRA A로 초기화한다.


    Args:
        model : DeBERTaV2 XXL
        approx_rank (int): Model의 Head 개수 | LoRA A, LoRA B의 Rank
    """
    len_of_layers = len(model.base_model.model.deberta.encoder.layer)
    for i in range(len_of_layers):
        q_loraA_head_list = []
        v_loraA_head_list = []

        q_weight = model.base_model.model.deberta.encoder.layer[i].attention.self.query_proj.weight.T
        v_weight = model.base_model.model.deberta.encoder.layer[i].attention.self.value_proj.weight.T

        all_q_head = copy.deepcopy(q_weight)
        all_q_head = all_q_head.reshape(approx_rank, 1536, 1536 // approx_rank)

        all_v_head = copy.deepcopy(v_weight)
        all_v_head = all_v_head.reshape(approx_rank, 1536, 1536 // approx_rank)

        for j in range(approx_rank):
            q_head = all_q_head[j]
            q_u, q_s, q_vt = torch.linalg.svd(q_head)
            q_loraA_head = q_u[:, :1] @ torch.diag(q_s[:1])

            v_head = all_v_head[j]
            v_u, v_s, v_vt = torch.linalg.svd(v_head)
            v_loraA_head = v_u[:, :1] @ torch.diag(v_s[:1])

            q_loraA_head_list.append(q_loraA_head)
            v_loraA_head_list.append(v_loraA_head)
        print(
            f"Complete Merge Head! len of q_loraA_head_list: {len(q_loraA_head_list)},len of v_loraA_head_list: {len(v_loraA_head_list)}"
        )
        q_loraA = torch.cat(q_loraA_head_list, dim=1)
        v_loraA = torch.cat(v_loraA_head_list, dim=1)

        model.base_model.model.deberta.encoder.layer[
            i
        ].attention.self.query_proj.lora_A.default.weight.data = q_loraA.T.contiguous()
        model.base_model.model.deberta.encoder.layer[
            i
        ].attention.self.value_proj.lora_A.default.weight.data = v_loraA.T.contiguous()

        model.base_model.model.deberta.encoder.layer[
            i
        ].attention.self.query_proj.lora_B.default.weight.data = torch.zeros_like(
            model.base_model.model.deberta.encoder.layer[i].attention.self.query_proj.lora_B.default.weight.data
        )
        model.base_model.model.deberta.encoder.layer[
            i
        ].attention.self.value_proj.lora_B.default.weight.data = torch.zeros_like(
            model.base_model.model.deberta.encoder.layer[i].attention.self.value_proj.lora_B.default.weight.data
        )

        print(f"Complete init LoraA! layer: {i}, q_loraA: {q_loraA.shape}, v_loraA: {v_loraA.shape}")


def deberta_init_dW_B_T_with_svd_us_by_head(model, approx_rank: int):
    """
    Pretrained W를 Head-Wise로 나눈 후
    각 Head를 SVD를 통해 분해 (U @ S @ Vt)해서 Head 마다 [Hidden_dim, 1]하나를 뽑아낸다.
    이를 Concat하여 LoRA A로 초기화한다.


    Args:
        model : DeBERTaV2 XXL
        approx_rank (int): Model의 Head 개수 | LoRA A, LoRA B의 Rank
    """
    len_of_layers = len(model.base_model.model.deberta.encoder.layer)
    for i in range(len_of_layers):
        q_loraB_head_list = []
        v_loraB_head_list = []

        q_weight = model.base_model.model.deberta.encoder.layer[i].attention.self.query_proj.weight.T
        v_weight = model.base_model.model.deberta.encoder.layer[i].attention.self.value_proj.weight.T

        all_q_head = copy.deepcopy(q_weight)
        all_q_head = all_q_head.reshape(approx_rank, 1536, 1536 // approx_rank)

        all_v_head = copy.deepcopy(v_weight)
        all_v_head = all_v_head.reshape(approx_rank, 1536, 1536 // approx_rank)

        for j in range(approx_rank):
            q_head = all_q_head[j]
            q_u, q_s, q_vt = torch.linalg.svd(q_head)
            q_loraB_head = q_u[:, :1] @ torch.diag(q_s[:1])

            v_head = all_v_head[j]
            v_u, v_s, v_vt = torch.linalg.svd(v_head)
            v_loraB_head = v_u[:, :1] @ torch.diag(v_s[:1])

            q_loraB_head_list.append(q_loraB_head)
            v_loraB_head_list.append(v_loraB_head)
        print(
            "-" * 50,
            f"Complete Merge Head! len of q_loraA_head_list: {len(q_loraB_head_list)},len of v_loraA_head_list: {len(v_loraB_head_list)}",
            "-" * 50,
        )
        q_loraB = torch.cat(q_loraB_head_list, dim=1)
        v_loraB = torch.cat(v_loraB_head_list, dim=1)

        model.base_model.model.deberta.encoder.layer[
            i
        ].attention.self.query_proj.lora_B.default.weight.data = q_loraB.contiguous()
        model.base_model.model.deberta.encoder.layer[
            i
        ].attention.self.value_proj.lora_B.default.weight.data = v_loraB.contiguous()

        model.base_model.model.deberta.encoder.layer[
            i
        ].attention.self.query_proj.lora_A.default.weight.data = torch.zeros_like(
            model.base_model.model.deberta.encoder.layer[i].attention.self.query_proj.lora_A.default.weight.data
        )
        model.base_model.model.deberta.encoder.layer[
            i
        ].attention.self.value_proj.lora_A.default.weight.data = torch.zeros_like(
            model.base_model.model.deberta.encoder.layer[i].attention.self.value_proj.lora_A.default.weight.data
        )

        print(f"Complete init LoraA! layer: {i}, q_loraA: {q_loraB.shape}, v_loraA: {v_loraB.shape}")


def deberta_init_dW_A_with_svd_us_by_head_scaling(model, approx_rank: int):
    len_of_layers = len(model.base_model.model.deberta.encoder.layer)
    for layer_idx in range(len_of_layers):
        print(f"{layer_idx}th Layer ")
        q_loraA_head_list = []
        v_loraA_head_list = []
        q_weight = model.base_model.model.deberta.encoder.layer[layer_idx].attention.self.query_proj.weight.T
        v_weight = model.base_model.model.deberta.encoder.layer[layer_idx].attention.self.value_proj.weight.T

        q_og_lora_A = model.base_model.model.deberta.encoder.layer[
            layer_idx
        ].attention.self.query_proj.lora_A.default.weight.T
        v_og_lora_A = model.base_model.model.deberta.encoder.layer[
            layer_idx
        ].attention.self.value_proj.lora_A.default.weight.T

        all_q_head = copy.deepcopy(q_weight)
        all_q_head = all_q_head.reshape(approx_rank, 1536, 1536 // approx_rank)
        all_v_head = copy.deepcopy(v_weight)
        all_v_head = all_v_head.reshape(approx_rank, 1536, 1536 // approx_rank)

        for j in range(approx_rank):
            q_head = all_q_head[j]
            q_u, q_s, q_vt = torch.linalg.svd(q_head)
            q_loraA_head = q_u[:, :1] @ torch.diag(q_s[:1])
            v_head = all_v_head[j]
            v_u, v_s, v_vt = torch.linalg.svd(v_head)
            v_loraA_head = v_u[:, :1] @ torch.diag(v_s[:1])
            print("-" * 50)
            print(
                f"{j}th head norm Before Scaling \n q_loraA_head : {torch.norm(q_loraA_head)} v_loraA_head : {torch.norm(v_loraA_head)}"
            )
            q_og_lora_A_icol_norm = torch.norm(q_og_lora_A[:, j : j + 1])
            v_og_lora_A_icol_norm = torch.norm(v_og_lora_A[:, j : j + 1])

            q_scale = q_og_lora_A_icol_norm / torch.norm(q_loraA_head)
            v_scale = v_og_lora_A_icol_norm / torch.norm(v_loraA_head)

            q_loraA_head = q_loraA_head * q_scale
            v_loraA_head = v_loraA_head * v_scale
            print(
                f"{j}th head norm After Scaling \n q_loraA_head : {torch.norm(q_loraA_head)} v_loraA_head : {torch.norm(v_loraA_head)}"
            )
            print("-" * 50)
            q_loraA_head_list.append(q_loraA_head)
            v_loraA_head_list.append(v_loraA_head)
        q_loraA = torch.cat(q_loraA_head_list, dim=1)
        v_loraA = torch.cat(v_loraA_head_list, dim=1)
        model.base_model.model.deberta.encoder.layer[
            layer_idx
        ].attention.self.query_proj.lora_A.default.weight.data = q_loraA.T.contiguous()
        model.base_model.model.deberta.encoder.layer[
            layer_idx
        ].attention.self.value_proj.lora_A.default.weight.data = v_loraA.T.contiguous()
        model.base_model.model.deberta.encoder.layer[
            layer_idx
        ].attention.self.query_proj.lora_B.default.weight.data = torch.zeros_like(
            model.base_model.model.deberta.encoder.layer[
                layer_idx
            ].attention.self.query_proj.lora_B.default.weight.data
        )
        model.base_model.model.deberta.encoder.layer[
            layer_idx
        ].attention.self.value_proj.lora_B.default.weight.data = torch.zeros_like(
            model.base_model.model.deberta.encoder.layer[
                layer_idx
            ].attention.self.value_proj.lora_B.default.weight.data
        )
        print(
            f"Complete init LoraA! layer: {layer_idx}, q_loraA: {torch.norm(model.base_model.model.deberta.encoder.layer[layer_idx].attention.self.query_proj.lora_A.default.weight)}, v_loraA: {torch.norm(model.base_model.model.deberta.encoder.layer[layer_idx].attention.self.value_proj.lora_A.default.weight)}"
        )

def deberta_init_dW_B_T_with_svd_us_by_head_scaling(model, approx_rank: int):
    len_of_layers = len(model.base_model.model.deberta.encoder.layer)
    for layer_idx in range(len_of_layers):
        print(f"{layer_idx}th Layer ")
        q_loraB_head_list = []
        v_loraB_head_list = []
        q_weight = model.base_model.model.deberta.encoder.layer[layer_idx].attention.self.query_proj.weight.T
        v_weight = model.base_model.model.deberta.encoder.layer[layer_idx].attention.self.value_proj.weight.T

        q_og_lora_A = model.base_model.model.deberta.encoder.layer[
            layer_idx
        ].attention.self.query_proj.lora_A.default.weight.T
        v_og_lora_A = model.base_model.model.deberta.encoder.layer[
            layer_idx
        ].attention.self.value_proj.lora_A.default.weight.T

        all_q_head = copy.deepcopy(q_weight)
        all_q_head = all_q_head.reshape(approx_rank, 1536, 1536 // approx_rank)
        all_v_head = copy.deepcopy(v_weight)
        all_v_head = all_v_head.reshape(approx_rank, 1536, 1536 // approx_rank)

        for j in range(approx_rank):
            q_head = all_q_head[j]
            q_u, q_s, q_vt = torch.linalg.svd(q_head)
            q_loraB_head = q_u[:, :1] @ torch.diag(q_s[:1])
            v_head = all_v_head[j]
            v_u, v_s, v_vt = torch.linalg.svd(v_head)
            v_loraB_head = v_u[:, :1] @ torch.diag(v_s[:1])
            print("-" * 50)
            print(
                f"{j}th head norm Before Scaling \n q_loraA_head : {torch.norm(q_loraB_head)} v_loraA_head : {torch.norm(v_loraB_head)}"
            )
            q_og_lora_A_icol_norm = torch.norm(q_og_lora_A[:, j : j + 1])
            v_og_lora_A_icol_norm = torch.norm(v_og_lora_A[:, j : j + 1])

            q_scale = q_og_lora_A_icol_norm / torch.norm(q_loraB_head)
            v_scale = v_og_lora_A_icol_norm / torch.norm(v_loraB_head)

            q_loraB_head = q_loraB_head * q_scale
            v_loraB_head = v_loraB_head * v_scale
            print(
                f"{j}th head norm After Scaling \n q_loraA_head : {torch.norm(q_loraB_head)} v_loraA_head : {torch.norm(v_loraB_head)}"
            )
            print("-" * 50)
            q_loraB_head_list.append(q_loraB_head)
            v_loraB_head_list.append(v_loraB_head)
            
        q_loraB = torch.cat(q_loraB_head_list, dim=1)
        v_loraB = torch.cat(v_loraB_head_list, dim=1)
        
        model.base_model.model.deberta.encoder.layer[
            layer_idx
        ].attention.self.query_proj.lora_B.default.weight.data = q_loraB.contiguous()
        model.base_model.model.deberta.encoder.layer[
            layer_idx
        ].attention.self.value_proj.lora_B.default.weight.data = v_loraB.contiguous()
        
        model.base_model.model.deberta.encoder.layer[
            layer_idx
        ].attention.self.query_proj.lora_A.default.weight.data = torch.zeros_like(
            model.base_model.model.deberta.encoder.layer[
                layer_idx
            ].attention.self.query_proj.lora_A.default.weight.data
        )
        model.base_model.model.deberta.encoder.layer[
            layer_idx
        ].attention.self.value_proj.lora_A.default.weight.data = torch.zeros_like(
            model.base_model.model.deberta.encoder.layer[
                layer_idx
            ].attention.self.value_proj.lora_A.default.weight.data
        )
        
        print(
            f"Complete init LoraB! layer: {layer_idx}, q_loraB: {torch.norm(model.base_model.model.deberta.encoder.layer[layer_idx].attention.self.query_proj.lora_B.default.weight)}, v_loraB: {torch.norm(model.base_model.model.deberta.encoder.layer[layer_idx].attention.self.value_proj.lora_B.default.weight)}"
        )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    apply_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply LoRA or not."},
    )
    lora_alpha: Optional[int] = field(
        default=None,
        metadata={"help": "LoRA alpha"},
    )
    lora_r: Optional[int] = field(
        default=None,
        metadata={"help": "LoRA r"},
    )
    lora_path: Optional[str] = field(
        default=None,
        metadata={"help": "The file path of LoRA parameters."},
    )
    apply_adapter: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply adapter or not."},
    )
    adapter_path: Optional[str] = field(
        default=None,
        metadata={"help": "The file path of adapter parameters."},
    )
    adapter_type: Optional[str] = field(
        default="houlsby",
        metadata={"help": "houlsby or pfeiffer"},
    )
    adapter_size: Optional[int] = field(
        default=64,
        metadata={"help": "8, 16, 32, 64"},
    )
    apply_bitfit: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply bitfit or not."},
    )
    reg_loss_wgt: Optional[float] = field(
        default=0.0,
        metadata={"help": "Regularization Loss Weight"},
    )
    masking_prob: Optional[float] = field(
        default=0.0,
        metadata={"help": "Token Masking Probability"},
    )
    ex_type: Optional[str] = field(
        default="DEFAULT",
        metadata={"help": "Experiment Type"},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # torch.use_deterministic_algorithms(training_args.use_deterministic_algorithms)
    # logger.info("use_deterministic_algorithms: " + str(torch.are_deterministic_algorithms_enabled()))

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue", data_args.task_name)
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            datasets = load_dataset("csv", data_files=data_files)
        else:
            # Loading a dataset from local json files
            datasets = load_dataset("json", data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        # cls_dropout=training_args.cls_dropout,
        apply_lora=model_args.apply_lora,
        lora_alpha=model_args.lora_alpha,
        lora_r=model_args.lora_r,
        apply_adapter=model_args.apply_adapter,
        adapter_type=model_args.adapter_type,
        adapter_size=model_args.adapter_size,
        reg_loss_wgt=model_args.reg_loss_wgt,
        masking_prob=model_args.masking_prob,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    """
    peft.LoraConfig는 Linear forward 시
    A = (r,d) B = (d,r) 이므로
    result += B(A(x)) 
    x @ A @ B
    (d) @ (d,r) @ (r,d)
    """
    config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules=["query_proj", "value_proj"],
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, config)

    """
    Initialize 세팅 별로 다른 함수 호출한다.
    """
    if "deberta_init_dW_A_with_svd_us_by_head" == model_args.ex_type:
        print("-" * 25, "deberta_init_dW_A_with_svd_us_by_head", "-" * 25)
        deberta_init_dW_A_with_svd_us_by_head(model, model_args.lora_r)
    elif "init_dW_A_with_svd_from_back_with_scaling_entire" == model_args.ex_type:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_og = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v2-xxlarge", num_labels=2)
        model_og.to(device)
        deberta_init_dW_A_with_svd_from_back_with_scaling_entire(model, model_og, model_args.lora_r)
        print("######init_dW_A_with_svd_from_back_with_scaling_entire#####")
    elif "scale_after_lora" == model_args.ex_type:
        print("######scale_after_lora#####")
        normal_peft_model_id = "/home/lab/bumjun/low_rank/examples/NLU/output/deberta_init_dW_A_with_svd_us_by_head-rank24-alpha24-seed0/model/checkpoint-800"  # "/home/lab/bumjun/low_rank/examples/NLU/output/normal-rank16/model/checkpoint-3400"
        normal_config = PeftConfig.from_pretrained(normal_peft_model_id)
        tokenizer = AutoTokenizer.from_pretrained(normal_config.base_model_name_or_path)

        model = AutoModelForSequenceClassification.from_pretrained(normal_config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, normal_peft_model_id)
        normal_rank16_norm_lora_A = {}
        normal_rank16_norm_lora_B = {}

        for name, param in model.named_parameters():
            if "lora_A" in name and "bias" not in name:
                norm_lora_A = param.data.norm()
                normal_rank16_norm_lora_A[name] = norm_lora_A.item()
            elif "lora_B" in name and "bias" not in name:
                norm_lora_B = param.data.norm()
                normal_rank16_norm_lora_B[name] = norm_lora_B.item()

        normal_rank16_norm_lora_A_df = pd.DataFrame.from_dict(
            normal_rank16_norm_lora_A, orient="index", columns=["norm"]
        )
        normal_rank16_norm_lora_B_df = pd.DataFrame.from_dict(
            normal_rank16_norm_lora_B, orient="index", columns=["norm"]
        )
        norm_A_divided_by_B = []
        norm_B_divided_by_A = []
        for i in range(len(normal_rank16_norm_lora_A_df)):
            norm_A_divided_by_B.append(
                np.sqrt(normal_rank16_norm_lora_A_df["norm"][i] / normal_rank16_norm_lora_B_df["norm"][i])
            )
            norm_B_divided_by_A.append(
                np.sqrt(normal_rank16_norm_lora_B_df["norm"][i] / normal_rank16_norm_lora_A_df["norm"][i])
            )

        for i in range(len(model.base_model.model.deberta.encoder.layer)):
            model.base_model.model.deberta.encoder.layer[i].attention.self.query_proj.lora_A.default.weight.data = (
                model.base_model.model.deberta.encoder.layer[i].attention.self.query_proj.lora_A.default.weight.data
                * norm_B_divided_by_A[2 * i]
            )
            model.base_model.model.deberta.encoder.layer[i].attention.self.value_proj.lora_A.default.weight.data = (
                model.base_model.model.deberta.encoder.layer[i].attention.self.value_proj.lora_A.default.weight.data
                * norm_B_divided_by_A[2 * i + 1]
            )

            model.base_model.model.deberta.encoder.layer[i].attention.self.query_proj.lora_B.default.weight.data = (
                model.base_model.model.deberta.encoder.layer[i].attention.self.query_proj.lora_B.default.weight.data
                * norm_A_divided_by_B[2 * i]
            )
            model.base_model.model.deberta.encoder.layer[i].attention.self.value_proj.lora_B.default.weight.data = (
                model.base_model.model.deberta.encoder.layer[i].attention.self.value_proj.lora_B.default.weight.data
                * norm_A_divided_by_B[2 * i + 1]
            )
        print(
            "######Norm After Scale#####",
            torch.norm(
                model.base_model.model.deberta.encoder.layer[0].attention.self.value_proj.lora_A.default.weight.data
            ),
            torch.norm(
                model.base_model.model.deberta.encoder.layer[0].attention.self.value_proj.lora_B.default.weight.data
            ),
        )
    elif "deberta_init_dW_A_with_svd_scaling" == model_args.ex_type:
        print("######deberta_init_dW_A_with_svd_scaling#####")
        print("\nSANITY CHECK\n")
        print(
            f"변경 전 LoRA A : {model.base_model.model.deberta.encoder.layer[0].attention.self.query_proj.lora_A.default.weight.data}"
        )
        deberta_init_dW_A_with_svd_scaling(model, model_args.lora_r)
        print(
            f"변경 후 LoRA A : {model.base_model.model.deberta.encoder.layer[0].attention.self.query_proj.lora_A.default.weight.data}"
        )
    elif "deberta_init_dW_with_svd" == model_args.ex_type:
        print("-" * 25, "deberta_init_dW_with_svd", "-" * 25)
        deberta_init_dW_with_svd(model, model_args.lora_r)
    elif "deberta_init_dW_A_scaling_with_svd_us_by_head" == model_args.ex_type:
        print("-" * 25, "deberta_init_dW_A_with_svd_us_by_head_scaling", "-" * 25)
        deberta_init_dW_A_with_svd_us_by_head_scaling(model, model_args.lora_r)
    elif "deberta_init_dW_with_svd_from_back" == model_args.ex_type:
        print("-" * 25, "deberta_init_dW_with_svd_from_back", "-" * 25)
        deberta_init_dW_with_svd_from_back(model, model_args.lora_r)
        print("-" * 50)
    elif "deberta_init_dW_A_with_svd" == model_args.ex_type:
        print("-" * 25, "deberta_init_dW_A_with_svd", "-" * 25)
        deberta_init_dW_A_with_svd(model, model_args.lora_r)
        print("-" * 50)
    elif "deberta_init_dW_B_T_with_svd_us_by_head" == model_args.ex_type:
        print("-" * 50, "deberta_init_dW_B_T_with_svd_us_by_head", "-" * 50)
        deberta_init_dW_B_T_with_svd_us_by_head(model, model_args.lora_r)
        print("-" * 100)
    elif "deberta_init_dW_B_T_with_svd_us_by_head_scaling" == model_args.ex_type:
        print("-" * 50, "deberta_init_dW_B_T_with_svd_us_by_head_scaling", "-" * 50)
        deberta_init_dW_B_T_with_svd_us_by_head_scaling(model, model_args.lora_r)
        print("-" * 100)
    else:
        print("-" * 25, "DEFAULT", "-" * 25)
        print("-" * 50)

    """
    Wandb에 Logging 하기 위해 Name 설정
    """
    os.environ["WANDB_NAME"] = (
        model_args.ex_type
        + "-rank"
        + str(model_args.lora_r)
        + "-alpha"
        + str(model_args.lora_alpha)
        + "-seed"
        + str(training_args.seed)
    )

    trainable_params = []
    if model_args.apply_lora:
        if model_args.lora_path is not None:
            # lora_state_dict = torch.load(model_args.lora_path)
            logger.info(f"Apply LoRA state dict from {model_args.lora_path}.")
            model.load_adapter(model_args.lora_path, adapter_name="lora")
            # logger.info(lora_state_dict.keys())
            # model.load_state_dict(lora_state_dict, strict=False)
        trainable_params.append("lora")

    if model_args.apply_adapter:
        if model_args.adapter_path is not None:
            adapter_state_dict = torch.load(os.path.join(model_args.adapter_path, "pytorch_adapter.bin"))
            head_state_dict = torch.load(os.path.join(model_args.adapter_path, "pytorch_model_head.bin"))
            added_state_dict = {}
            for k, v in adapter_state_dict.items():
                new_k = (
                    k.replace(data_args.task_name + ".", "")
                    .replace("adapter_down.0.", "adapter_A.")
                    .replace("adapter_up.", "adapter_B.")
                    .replace(".adapters.", ".adapter.")
                )
                added_state_dict[new_k] = v
            for k, v in head_state_dict.items():
                new_k = k.replace("heads." + data_args.task_name + ".1", "classifier.dense").replace(
                    "heads." + data_args.task_name + ".4", "classifier.out_proj"
                )
                added_state_dict[new_k] = v
            logger.info(f"Apply adapter state dict from {model_args.adapter_path}.")
            logger.info(added_state_dict.keys())
            missing_keys, unexpected_keys = model.load_state_dict(added_state_dict, strict=False)
            for missing_key in missing_keys:
                assert "adapter" not in missing_key, missing_key + " is missed in the model"
            assert len(unexpected_keys) == 0, "Unexpected keys " + str(unexpected_keys)
        trainable_params.append("adapter")

    if model_args.apply_bitfit:
        trainable_params.append("bias")

    # lora 안붙은 layer는 학습 안되게 하기
    if len(trainable_params) > 0:
        for name, param in model.named_parameters():
            if name.startswith("deberta") or name.startswith("base_model"):
                param.requires_grad = False
                for trainable_param in trainable_params:
                    if trainable_param in name:
                        param.requires_grad = True
                        break
            else:
                param.requires_grad = True
    print_trainable_parameters(model)

    # Preprocessing the datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)

    # training_args.metric_for_best_model = "accuracy"
    training_args.load_best_model_at_end = True
    # training_args.greater_is_better = True

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in datasets and "validation_matched" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in datasets and "test_matched" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)
    # TODO: When datasets metrics include regular accuracy, make an else here and remove special branch from
    # compute_metrics

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    print(model)
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=15)],
    )

    # USE CUSTOM TRAINER
    # trainer = CustomTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset if training_args.do_train else None,
    #     eval_dataset=eval_dataset if training_args.do_eval else None,
    #     compute_metrics=compute_metrics,
    #     tokenizer=tokenizer,
    #     data_collator=data_collator,
    #     callbacks=[EarlyStoppingCallback(early_stopping_patience=15)],
    # )

    # Training
    if training_args.do_train:
        checkpoint = None
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            # Check the config from that potential checkpoint has the right number of labels before using it as a
            # checkpoint.
            if AutoConfig.from_pretrained(model_args.model_name_or_path).num_labels == num_labels:
                checkpoint = model_args.model_name_or_path

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
            metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Test ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            test_datasets.append(datasets["test_mismatched"])

        for test_dataset, task in zip(test_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            test_dataset.remove_columns_("label")
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_test_file = os.path.join(training_args.output_dir, f"test_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info(f"***** Test results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
