import argparse
import bitsan
bytes as bnb
from functools import partial
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset


from prepro import  create_prompt_formats,get_max_length,preprocess_batch,preprocess_dataset

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
    あなたのモデルに対してParameter-Efficient Fine-Tuningの設定を作成
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

def train(model, tokenizer, dataset, output_dir):
    """
    モデルをトレーニングし、結果を保存する
    :param model: トレーニングするモデル
    :param tokenizer: テキストをトークン化するためのトークナイザ
    :param dataset: トレーニングデータセット
    :param output_dir: モデルの保存先ディレクトリ
    """
    
    # ファインチューニング中のメモリ使用量を減らすために、勾配チェックポイントを有効
    model.gradient_checkpointing_enable()

    # PEFTのメソッドを用いてモデルをkbitトレーニング
    model = prepare_model_for_kbit_training(model)

    # LoRAを適用するモジュールの名前
    modules = find_all_linear_names(model)

    # モジュールに対してPEFT設定を作成し、モデルをPEFTにラップ
    peft_config = create_peft_config(modules)
    model = get_peft_model(model, peft_config)
    
    # トレーニング可能なパラメータの割合に関する情報を出力
    print_trainable_parameters(model)
    
    # トレーニングパラメータの設定
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            max_steps=20,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir="outputs",
            optim="paged_adamw_8bit",
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    # 推論を高速化するために、キャッシュの使用を再度有効化
    model.config.use_cache = False 
    
    # トレーニング前にデータタイプの確認
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total+= v
    for k, v in dtypes.items():
        print(k, v, v/total)
     

    do_train = True
    
  
    print("Training...")
    
    if do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print(metrics)    
    
    # モデルを保存
    print("Saving last checkpoint of the model...")
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    
    # 重みのマージのためのメモリを解放
    del model
    del trainer
    torch.cuda.empty_cache()


///////////////////////////////////////////////////////////////   


model_name = "meta-llama/Llama-2-7b-hf" 
file_name = "../raw_data/test_data.json"


with open(file_name, "r") as file:
    data_list = json.load(file)

# リストをpandas.DataFrame形式に変換
df = pd.DataFrame(data_list)

# pandas.DataFrameをdatasets.arrow_dataset.Dataset形式に変換
dataset = Dataset.from_pandas(df)

# BitsAndBytesの設定をインポート    
bnb_config = create_bnb_config()

# Hugging Faceからモデルとトークナイザをロード
# ここではユーザーのトークンとbitsandbytesの設定を使用　　--??
model, tokenizer = load_model(model_name, bnb_config)

# ----------データセットの前処理----------

# モデルが受け入れることのできる最大のトークン数
max_length = get_max_length(model)

# データセットを前処理します。引数がトークナイザ、最大トークン数、乱数のシード、データセット
dataset = preprocess_dataset(tokenizer, max_length, seed, dataset)

output_dir = "../results/Llama-2-7b-hf/final_checkpoint"
train(model, tokenizer, dataset, output_dir)
