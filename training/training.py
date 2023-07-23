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

    return bnb_config

def create_peft_config(modules):
    """
    あなたのモデルに対してParameter-Efficient Fine-Tuningの設定を作成します
    :param modules: Loraを適用するモジュールの名前
    """
    # LoRAの設定を作成
    config = LoraConfig(
        r=16,  # 更新される行列の次元
        lora_alpha=64,  # スケーリングパラメータ
        target_modules=modules,  # LoRAを適用する対象のモジュール
        lora_dropout=0.1,  # レイヤーのドロップアウト確率
        bias="none",  # バイアス項の設定
        task_type="CAUSAL_LM",  # タスクの種類 現在- 因果言語モデル
    )

  
    return config

def find_all_linear_names(model):
    """
    --モデル内の全ての線形モジュール（torch.nn.Linear, bnb.nn.Linear4bitなど）の名前を見つける--
    :param model: モジュール名を探す対象のモデル
    """
    
    # 線形モジュールとしてbnb.nn.Linear4bitを指定
    cls = bnb.nn.Linear4bit

 
    lora_module_names = set()

    # モデル内の全てのモジュールをだす
    for name, module in model.named_modules():
        # もしモジュールが指定した線形モジュールのインスタンスならその名前を保存する
        if isinstance(module, cls):
            # '.'で分割して最初か最後の名前を取得
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    # もし'lm_head'がモジュール名に含まれていたら削除　これは、16ビットの設定で必要な処理らしい
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')

    # 線形モジュールの名前のリスト
    return list(lora_module_names)
