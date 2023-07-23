import argparse
import bitsandbytes as bnb
from functools import partial
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset

def load_model(model_name, bnb_config):
    n_gpus = torch.cuda.device_count()
    max_memory = f'{40960}MB'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto", # dispatch efficiently the model on the available ressources
        max_memory = {i: max_memory for i in range(n_gpus)},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)


    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def create_bnb_config():
    #https://huggingface.co/docs/transformers/main_classes/quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, # 4ビットでデータをロードする
        bnb_4bit_use_double_quant=True, # 4ビットのデータで量子化を二回行う
        bnb_4bit_quant_type="nf4", # 4ビットの量子化タイプは 'nf4' を指定
        bnb_4bit_compute_dtype=torch.bfloat16, # 4ビットデータの演算で使用するデータ型は 'torch.bfloat16' を指定
    )

    # 作成した設定を返す
    return bnb_config

