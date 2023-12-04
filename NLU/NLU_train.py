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
    EvalPrediction,
    PretrainedConfig,
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
    PeftModel,
)
from collections import Counter
import glob
import time
import datasets
from datasets import load_dataset
from utils import compute_metrics, print_trainable_parameters
import loralib as lora

MODEL_SAVE_REPO = "MayIBorn/cola-deberta_initialize_dW_with_span-rank16"
HUGGINGFACE_AUTH_TOKEN = "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"

os.environ["WANDB_PROJECT"] = "DeBERTaV2_CoLA"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore
    torch.use_deterministic_algorithms(True)


seed_everything(42)


def add_lora_to_deberta(model, dim, rank, lora_alpha):
    len_of_layers = len(model.deberta.encoder.layer)
    for i in range(len_of_layers):
        model.deberta.encoder.layer[i].attention.self.query_proj = copy.deepcopy(
            lora.Linear(dim, dim, r=rank, lora_alpha=lora_alpha, merge_weights=False)
        )
        model.deberta.encoder.layer[i].attention.self.value_proj = copy.deepcopy(
            lora.Linear(dim, dim, r=rank, lora_alpha=lora_alpha, merge_weights=False)
        )


def copy_weights_to_deberta(model, W_model):
    """
    W_model의 W weight를 new_model의 W weight로 복사
    """
    len_of_layers = len(model.deberta.encoder.layer)
    q_encoder_weight_list = []
    v_encoder_weight_list = []
    q_encoder_bias_list = []
    v_encoder_bias_list = []

    for i in range(len_of_layers):
        q_encoder_new_weight = W_model.deberta.encoder.layer[i].attention.self.query_proj.weight.data
        q_encoder_weight_list.append(q_encoder_new_weight)
        q_encoder_new_bias = W_model.deberta.encoder.layer[i].attention.self.query_proj.bias.data
        q_encoder_bias_list.append(q_encoder_new_bias)

        v_encoder_new_weight = W_model.deberta.encoder.layer[i].attention.self.value_proj.weight.data
        v_encoder_weight_list.append(v_encoder_new_weight)
        v_encoder_new_bias = W_model.deberta.encoder.layer[i].attention.self.value_proj.bias.data
        v_encoder_bias_list.append(v_encoder_new_bias)

    with torch.no_grad():
        for i in range(len_of_layers):
            model.deberta.encoder.layer[i].attention.self.query_proj.weight.data.copy_(q_encoder_weight_list[i])
            model.deberta.encoder.layer[i].attention.self.value_proj.weight.data.copy_(v_encoder_weight_list[i])
            model.deberta.encoder.layer[i].attention.self.query_proj.bias.data.copy_(q_encoder_bias_list[i])
            model.deberta.encoder.layer[i].attention.self.value_proj.bias.data.copy_(v_encoder_bias_list[i])


def deberta_initialize_dW_with_svd(model, model_original, approx_rank):
    q_proj_v_loraA_weights = []
    k_proj_v_loraA_weights = []
    q_proj_v_loraB_weights = []
    k_proj_v_loraB_weights = []

    len_of_layers = len(model_original.deberta.encoder.layer)
    with torch.no_grad():
        for i in range(len_of_layers):
            q_original_weight = model_original.deberta.encoder.layer[i].attention.self.query_proj.weight.data.T
            k_original_weight = model_original.deberta.encoder.layer[i].attention.self.value_proj.weight.data.T

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


def deberta_initialize_dW_A_with_svd(model, model_original, approx_rank):
    q_proj_v_loraA_weights = []
    k_proj_v_loraA_weights = []

    len_of_layers = len(model_original.deberta.encoder.layer)
    with torch.no_grad():
        for i in range(len_of_layers):
            q_original_weight = model_original.deberta.encoder.layer[i].attention.self.query_proj.weight.data.T
            k_original_weight = model_original.deberta.encoder.layer[i].attention.self.value_proj.weight.data.T

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


def initialize_dW_A_with_svd_inv(model, model_original, approx_rank):
    q_proj_v_loraA_weights = []
    k_proj_v_loraA_weights = []

    len_of_layers = len(model_original.model.layers)
    with torch.no_grad():
        for i in range(len_of_layers):
            q_original_weight = model_original.model.layers[i].self_attn.q_proj.weight.data.T
            k_original_weight = model_original.model.layers[i].self_attn.k_proj.weight.data.T

            q_inv_weight = torch.linalg.inv(q_original_weight)
            k_inv_weight = torch.linalg.inv(k_original_weight)

            q_proj_u, q_proj_s, q_proj_v = torch.linalg.svd(q_inv_weight)
            k_proj_u, k_proj_s, k_proj_v = torch.linalg.svd(k_inv_weight)

            # w_q_encoder

            q_proj_v_loraA_weights.append(q_proj_u[:, :approx_rank] @ torch.diag(q_proj_s[:approx_rank]).sqrt())

            # w_v_encoder
            k_proj_v_loraA_weights.append(k_proj_u[:, :approx_rank] @ torch.diag(k_proj_s[:approx_rank]).sqrt())

    with torch.no_grad():
        for i in range(len_of_layers):
            model.base_model.model.model.layers[i].self_attn.q_proj.lora_A.default.weight.data = copy.deepcopy(
                q_proj_v_loraA_weights[i].transpose(0, 1).contiguous()
            )

            model.base_model.model.model.layers[i].self_attn.k_proj.lora_A.default.weight.data = copy.deepcopy(
                k_proj_v_loraA_weights[i].transpose(0, 1).contiguous()
            )


def deberta_initialize_dW_B_with_svd(model, model_original, approx_rank):
    q_proj_v_loraB_weights = []
    k_proj_v_loraB_weights = []

    len_of_layers = len(model_original.deberta.encoder.layer)
    with torch.no_grad():
        for i in range(len_of_layers):
            q_original_weight = model_original.deberta.encoder.layer[i].attention.self.query_proj.weight.data.T
            k_original_weight = model_original.deberta.encoder.layer[i].attention.self.value_proj.weight.data.T

            q_proj_u, q_proj_s, q_proj_v = torch.linalg.svd(q_original_weight)
            k_proj_u, k_proj_s, k_proj_v = torch.linalg.svd(k_original_weight)

            # torch.Size([rank, 768])
            q_proj_v_loraB_weights.append(torch.diag(q_proj_s[:approx_rank]).sqrt() @ q_proj_v[:approx_rank, :])

            k_proj_v_loraB_weights.append(torch.diag(k_proj_s[:approx_rank]).sqrt() @ k_proj_v[:approx_rank, :])

    with torch.no_grad():
        for i in range(len_of_layers):
            model.deberta.encoder.layer[i].attention.self.query_proj.lora_B.default.weight.data = copy.deepcopy(
                q_proj_v_loraB_weights[i].transpose(0, 1).contiguous()
            )

            model.deberta.encoder.layer[i].attention.self.value_proj.lora_B.default.weight.data = copy.deepcopy(
                k_proj_v_loraB_weights[i].transpose(0, 1).contiguous()
            )


def deberta_initialize_dW_with_svd_from_back(model, model_original, approx_rank):
    q_proj_v_loraB_weights = []
    k_proj_v_loraB_weights = []
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


def deberta_initialize_dW_A_with_svd_from_back(model, model_original, approx_rank):
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


def deberta_initialize_dW_B_with_svd_from_back(model, model_original, approx_rank):
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


def deberta_initialize_dW_B_with_svd_from_back_dW_A_zero(model, model_original, approx_rank):
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


def deberta_initialize_dW_A_with_svd_selectively_from_back(model, model_original, approx_rank):
    q_proj_v_loraA_weights = []
    k_proj_v_loraA_weights = []
    len_of_layers = len(model_original.deberta.encoder.layer)

    with torch.no_grad():
        for i in range(len_of_layers):
            q_original_weight = model_original.deberta.encoder.layer[i].attention.self.query_proj.weight.data.T
            k_original_weight = model_original.deberta.encoder.layer[i].attention.self.value_proj.weight.data.T

            q_proj_u, q_proj_s, q_proj_v = torch.linalg.svd(q_original_weight)
            k_proj_u, k_proj_s, k_proj_v = torch.linalg.svd(k_original_weight)

            q_proj_v_loraA_weights.append(
                torch.cat((q_proj_u[:, :2], q_proj_u[:, -approx_rank + 2 :]), dim=1)
                @ torch.diag(torch.cat((q_proj_s[:2], q_proj_s[-approx_rank + 2 :])).sqrt())
            )

            k_proj_v_loraA_weights.append(
                torch.cat((k_proj_u[:, :2], k_proj_u[:, -approx_rank + 2 :]), dim=1)
                @ torch.diag(torch.cat((k_proj_s[:2], k_proj_s[-approx_rank + 2 :])).sqrt())
            )

    with torch.no_grad():
        for i in range(len_of_layers):
            model.deberta.encoder.layer[i].attention.self.query_proj.lora_A.default.weight.data = copy.deepcopy(
                q_proj_v_loraA_weights[i].transpose(0, 1).contiguous()
            )
            model.deberta.encoder.layer[i].attention.self.value_proj.lora_A.default.weight.data = copy.deepcopy(
                k_proj_v_loraA_weights[i].transpose(0, 1).contiguous()
            )


def deberta_initialize_dW_A_with_svd_from_back_with_scaling(model, model_original, approx_rank):
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


def deberta_initialize_dW_A_with_svd_from_back_with_scaling_entire(model, model_original, approx_rank):
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


def deberta_initialize_dW_B_with_svd_from_back_with_scaling_entire(model, model_original, approx_rank):
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
            model.deberta.encoder.layer[i].attention.self.query_proj.lora_B.default.weight.data = copy.deepcopy(
                q_proj_v_loraB_weights[i].transpose(0, 1).contiguous()
            )
            model.deberta.encoder.layer[i].attention.self.value_proj.lora_B.default.weight.data = copy.deepcopy(
                k_proj_v_loraB_weights[i].transpose(0, 1).contiguous()
            )


def initialize_dW_with_svd_from_back_with_scaling_A_Only(model, model_original, approx_rank):
    q_proj_v_loraA_weights = []
    k_proj_v_loraA_weights = []
    q_proj_v_loraB_weights = []
    k_proj_v_loraB_weights = []
    len_of_layers = len(model_original.model.layers)

    with torch.no_grad():
        for i in range(len_of_layers):
            q_numerator = 0
            q_denominator = 0
            k_numerator = 0
            k_denominator = 0
            q_scaling = 0
            k_scaling = 0

            q_original_weight = model_original.model.layers[i].self_attn.q_proj.weight.data.T
            k_original_weight = model_original.model.layers[i].self_attn.k_proj.weight.data.T

            q_proj_u, q_proj_s, q_proj_v = torch.linalg.svd(q_original_weight)
            k_proj_u, k_proj_s, k_proj_v = torch.linalg.svd(k_original_weight)
            q_proj_s_cpu = q_proj_s.cpu().numpy()
            k_proj_s_cpu = k_proj_s.cpu().numpy()
            for i in range(approx_rank):
                print(f"{i}번째 singular value: {q_proj_s_cpu[i]}")
                print(f"{-(i+1)}번째 singular value: {q_proj_s_cpu[-(i+1)]}")
                q_numerator += np.power(q_proj_s_cpu[i], 2)
                q_denominator += np.power(q_proj_s_cpu[-(i + 1)], 2)

                k_numerator += np.power(k_proj_s_cpu[i], 2)
                k_denominator += np.power(k_proj_s_cpu[-(i + 1)], 2)

            q_scaling = np.sqrt(q_numerator / q_denominator)
            k_scaling = np.sqrt(k_numerator / k_denominator)
            print(f"q_scaling: {q_scaling}")
            print(f"k_scaling: {k_scaling}")

            # Do not scaling B
            q_proj_v_loraB_weights.append(torch.diag(q_proj_s[-approx_rank:]).sqrt() @ q_proj_v[-approx_rank:, :])
            k_proj_v_loraB_weights.append(torch.diag(k_proj_s[-approx_rank:]).sqrt() @ k_proj_v[-approx_rank:, :])

            q_proj_s *= q_scaling
            k_proj_s *= k_scaling
            q_proj_v_loraA_weights.append(q_proj_u[:, -approx_rank:] @ torch.diag(q_proj_s[-approx_rank:]).sqrt())
            k_proj_v_loraA_weights.append(k_proj_u[:, -approx_rank:] @ torch.diag(k_proj_s[-approx_rank:]).sqrt())

    with torch.no_grad():
        for i in range(len_of_layers):
            model.base_model.model.model.layers[i].self_attn.q_proj.lora_A.default.weight.data = copy.deepcopy(
                q_proj_v_loraA_weights[i].transpose(0, 1).contiguous()
            )

            model.base_model.model.model.layers[i].self_attn.k_proj.lora_A.default.weight.data = copy.deepcopy(
                k_proj_v_loraA_weights[i].transpose(0, 1).contiguous()
            )

            model.base_model.model.model.layers[i].self_attn.q_proj.lora_B.default.weight.data = copy.deepcopy(
                q_proj_v_loraB_weights[i].transpose(0, 1).contiguous()
            )

            model.base_model.model.model.layers[i].self_attn.k_proj.lora_B.default.weight.data = copy.deepcopy(
                k_proj_v_loraB_weights[i].transpose(0, 1).contiguous()
            )


def deberta_initialize_dW_with_span(model, model_original, approx_rank):
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
        for forA in range(16):
            q_tilde_A = torch.zeros(1536)
            v_tilde_A = torch.zeros(1536)

            for j in range(forA, 16):
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


def deberta_initialize_dW_B_with_svd_by_head(model, approx_rank):
    len_of_layers = len(model.base_model.model.deberta.encoder.layer)
    for i in range(len_of_layers):
        q_loraB_head_list = []
        v_loraB_head_list = []

        q_weight = model.base_model.model.deberta.encoder.layer[i].attention.self.query_proj.weight
        v_weight = model.base_model.model.deberta.encoder.layer[i].attention.self.value_proj.weight

        all_q_head = copy.deepcopy(q_weight)
        all_q_head = all_q_head.view(24, 1536, 64)

        all_v_head = copy.deepcopy(v_weight)
        all_v_head = all_v_head.view(24, 1536, 64)
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

        model.base_model.model.deberta.encoder.layer[
            i
        ].attention.self.query_proj.lora_B.default.weight.data = q_loraB.T.contiguous()
        model.base_model.model.deberta.encoder.layer[
            i
        ].attention.self.value_proj.lora_B.default.weight.data = v_loraB.T.contiguous()

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


def train():
    task_name = "cola"
    max_seq_length = 64
    model_id = "microsoft/deberta-v2-xxlarge"
    datetime = time.strftime("%Y_%m_%d_%H_%M", time.localtime(time.time()))
    output_dir = f"./NLU/{datetime}"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = datasets.load_dataset("glue", f"{task_name}")
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

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
    config = LoraConfig(r=16, lora_alpha=32, target_modules=["query_proj", "value_proj"], task_type="SEQ_CLS")
    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    if task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in dataset["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    padding = "max_length"
    label_list = dataset["train"].features["label"].names
    num_labels = len(label_list)
    label_to_id = None
    if model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id and task_name is not None:
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        # else:
        #     logger.warn(
        #         "Your model seems to have been trained with labels, but they don't match the dataset: ",
        #         f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
        #         "\nIgnoring the model labels as a result.",
        #     )
    elif task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    max_seq_length = min(max_seq_length, tokenizer.model_max_length)

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

    dataset_ = dataset.map(preprocess_function, batched=True, load_from_cache_file=not False)

    dataset = load_dataset("glue", "cola")

    # 데이터셋 전처리
    def encode(example):
        return tokenizer(example["sentence"], truncation=True, padding="max_length")

    encoded_dataset = dataset.map(encode, batched=True, load_from_cache_file=not False)
    train_dataset = dataset_["train"]
    val_dataset = dataset_["validation"]
    test_dataset = dataset_["test"]

    # QLORA
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )
    # model = AutoModelForSequenceClassification.from_pretrained(model_id, quantization_config=bnb_config, num_labels=2)
    # model.gradient_checkpointing_enable()
    # model = prepare_model_for_kbit_training(model)

    model_og = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
    model_og.to(device)
    # model.to(device)
    # model.resize_token_embeddings(128000)
    # deberta_initialize_dW_B_with_svd_from_back_dW_A_zero(model, model_og, 16)
    # deberta_initialize_dW_B_with_svd_from_back_dW_A_zero(model, model_og, 16)
    # deberta_initialize_dW_B_with_svd_by_head(model, 16)
    deberta_initialize_dW_with_span(model, model_og, 16)
    # deberta_initialize_dW_with_span(model, model_og, 16)
    training_args = TrainingArguments(
        output_dir=f"{output_dir}",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        num_train_epochs=10,
        learning_rate=1.3e-4,
        weight_decay=0,
        fp16=True,
        do_eval=True,
        do_train=True,
        seed=42,
        save_total_limit=1,
        report_to="wandb",
        run_name=f"{task_name}-deberta_initialize_dW_with_span-rank16-fp16-{datetime}",
        logging_steps=1,
        logging_dir=f"{output_dir}/logs",
        push_to_hub=True,
        evaluation_strategy="steps",
        eval_steps=20,
        save_steps=40,
        save_strategy="steps",
        load_best_model_at_end=True,
        greater_is_better=True,
        metric_for_best_model="matthews_correlation",
        # use_deterministic_algorithms=True,
        # cls_dropout=0.1,
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
        # data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    # model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()
    model.push_to_hub(MODEL_SAVE_REPO, use_temp_dir=True)
    tokenizer.push_to_hub(MODEL_SAVE_REPO, use_temp_dir=True)


def evaluate():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datetime = time.strftime("%Y_%m_%d_%H_%M", time.localtime(time.time()))

    task_name = "cola"
    dataset = datasets.load_dataset("glue", f"{task_name}")
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
    max_seq_length = 64
    model_id = "microsoft/deberta-v2-xxlarge"
    output_dir = f"./NLU/{datetime}"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
    config = LoraConfig(r=16, lora_alpha=32, target_modules=["query_proj", "value_proj"], task_type="SEQ_CLS")
    model = get_peft_model(model, config)
    print_trainable_parameters(model)
    if task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in dataset["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    padding = "max_length"
    label_list = dataset["train"].features["label"].names
    num_labels = len(label_list)
    label_to_id = None
    if model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id and task_name is not None:
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        # else:
        #     logger.warn(
        #         "Your model seems to have been trained with labels, but they don't match the dataset: ",
        #         f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
        #         "\nIgnoring the model labels as a result.",
        #     )
    elif task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    max_seq_length = min(max_seq_length, tokenizer.model_max_length)

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

    dataset = dataset.map(preprocess_function, batched=True, load_from_cache_file=not False)
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
    # config = LoraConfig(r=16, lora_alpha=32, target_modules=["query_proj", "value_proj"], task_type="SEQ_CLS")
    # model = get_peft_model(model, config)
    peft_model_id = "/home/lab/bumjun/init/NLU/2023_11_27_20_58/checkpoint-1600"
    model = PeftModel.from_pretrained(model, peft_model_id)
    print_trainable_parameters(model)

    training_args = TrainingArguments(
        output_dir=f"{output_dir}",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        num_train_epochs=10,
        learning_rate=3e-4,
        weight_decay=0,
        fp16=True,
        do_eval=True,
        do_train=True,
        seed=42,
        save_total_limit=1,
        logging_steps=1,
        logging_dir=f"{output_dir}/logs",
        push_to_hub=True,
        evaluation_strategy="steps",
        eval_steps=40,
        save_steps=40,
        save_strategy="steps",
        load_best_model_at_end=True,
        greater_is_better=True,
        metric_for_best_model="matthews_correlation",
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        args=training_args,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        # data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False

    trainer.evaluate()


def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datetime = time.strftime("%Y_%m_%d_%H_%M", time.localtime(time.time()))

    task_name = "cola"
    dataset = datasets.load_dataset("glue", f"{task_name}")
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
    max_seq_length = 64
    model_id = "microsoft/deberta-v2-xxlarge"
    output_dir = f"./NLU/{datetime}"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    if task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in dataset["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    padding = "max_length"
    label_list = dataset["train"].features["label"].names
    num_labels = len(label_list)
    label_to_id = None
    if model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id and task_name is not None:
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        # else:
        #     logger.warn(
        #         "Your model seems to have been trained with labels, but they don't match the dataset: ",
        #         f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
        #         "\nIgnoring the model labels as a result.",
        #     )
    elif task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    max_seq_length = min(max_seq_length, tokenizer.model_max_length)

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

    dataset = dataset.map(preprocess_function, batched=True, load_from_cache_file=not False)

    test_dataset = dataset["test"]
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
    config = LoraConfig(r=16, lora_alpha=32, target_modules=["query_proj", "value_proj"], task_type="SEQ_CLS")
    model = get_peft_model(model, config)
    print_trainable_parameters(model)


def main():
    train()
    # train()
    # evaluate()


if __name__ == "__main__":
    main()
