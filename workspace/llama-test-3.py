from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    #Bitsandbytes==0.40.2じゃないと動作不可
    BitsAndBytesConfig,
)
import torch
import readline

name = "Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(name) #nameで指定したディレクトリ内のトークナイザーを読込む

tokenizer.pad_token_id = tokenizer.eos_token_id


#===== 量子化 =====
bnb_config = BitsAndBytesConfig(
    Load_in_4bit=True,   #4bit量子化を有効化
    bnb_4bit_quant_type="nf4",  #量子化のデータタイプを指定。4-bit NormalFloat Quantization のデータ型
    bnb_4bit_compute_dtype=torch.float16,   #量子化の演算時におけるデータタイプの指定
    bnb_4bit_use_double_quant=True, #ネストされた量子化を有効化
)
#==================

model = AutoModelForCausalLM.from_pretrained(
    name,
    quantization_config=bnb_config, #量子化設定を読み込み
    device_map="auto",  #使用するGPUを指定。今回の場合、全てのGPUを指定
    trust_remote_code=True, 
)
generation_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=True,
    device_map="auto",    # finds GPU
)

#sample_text = "ubuntu上でHDDのファイルシステムのUUIDを表示させるには？"    # prompt goes here
#sample_texttext = "ドクターペッパーとは何ですか？"
#sample_text = "[badblocks -snwf /dev/sdb]このコマンドが持つ意味を日本語で答えなさい。"


print("\n##################\n### Task Start ###\n##################\n")
while True:
    text = input("Please input >> ")
    if text=="exit":
        break

    sequences = generation_pipe(
        text,
        max_length=256,    #
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        #do_sample=True,
        top_k=40,
        #temperature=0.4,
        top_p=0.95
    )

    print(sequences[0]["generated_text"])

print("\n#####################\n### Task finished ###\n#####################\n")