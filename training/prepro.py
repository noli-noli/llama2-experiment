
# exsample...
# from datasets import load_dataset

# dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

# print(f"Number of prompts: {len(dataset)}")
# print(f"Column names are: {dataset.column_names}")
# print(f"head 3 : {dataset[1]}")

def create_prompt_formats(sample):
    """
    Format various fields of the sample ('instruction', 'context', 'response')
    Then concatenate them using two newline characters 
    :param sample: Sample dictionnary
    """

    INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    INSTRUCTION_KEY = "### Instruction:"
    INPUT_KEY = "Input:"
    RESPONSE_KEY = "### Response:"
    END_KEY = "### End"
    
    blurb = f"{INTRO_BLURB}"
    instruction = f"{INSTRUCTION_KEY}\n{sample['instruction']}"
    input_context = f"{INPUT_KEY}\n{sample['context']}" if sample["context"] else None
    response = f"{RESPONSE_KEY}\n{sample["answer"]}"
    end = f"{END_KEY}"
    
    parts = [part for part in [blurb, instruction, input_context, response, end] if part]

    formatted_prompt = "\n\n".join(parts)
    
    sample["text"] = formatted_prompt

    return sample

# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def get_max_length(model):
    # モデルの設定を取得
    conf = model.config
    max_length = None
    
    # モデルの設定から、シーケンスの最大長を取得
    # "n_positions", "max_position_embeddings", "seq_length" の順にチェック
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        # getattrを用いて指定した属性がある場合はその値を、ない場合はNoneを取得
        max_length = getattr(model.config, length_setting, None)
        
        # 最大長が取得できたらその値を表示し、ループを抜ける
        if max_length:
            print(f"Found max length: {max_length}")
            break
    
    # 上記のループで最大長が取得できなかった場合は、デフォルト値として1024を設定
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    
    # 最大長を返す
    return max_length


def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizing a batch
    """
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )


# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed, dataset: str):
    """データセットをトレーニングのための形式に整形・トークン化する
    :param tokenizer (AutoTokenizer): モデルのトークナイザ
    :param max_length (int): トークナイザから出力するトークンの最大数
    :param seed: データセットをシャッフルするためのシード
    :param dataset: 前処理を行うデータセット
    """
    
    # 各サンプルにプロンプトを追加
    print("Preprocessing dataset...")
    dataset = dataset.map(create_prompt_formats) # mapメソッドはデータセットの各要素に関数を適用します
    
    # データセットの各バッチに対して前処理を適用し、'instruction', 'context', 'response', 'text', 'category' フィールドを削除
    # partial関数は、関数とその一部の引数を取り、新しい関数を作成します
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True, # batched=Trueにすると、関数は全体のバッチに対して適用されます
        remove_columns=["instruction", "context", "response", "text", "category"], # これらの列は削除されます
    )

    # 'input_ids'が最大長を超えるサンプルをフィルタリングする
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)
    
    # データセットをシャッフル
    dataset = dataset.shuffle(seed=seed) # シードを使ってデータセットをシャッフルします

    # 前処理されたデータセットを返す
    return dataset

