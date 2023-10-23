from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    #Bitsandbytes==0.40.2じゃないと動作不可
    BitsAndBytesConfig,
)
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import torch


app = FastAPI()


model = "../models/Llama-2-13b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model) #modelで指定したディレクトリ内のトークナイザーを読込む
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
    model,
    #quantization_config=bnb_config, #量子化設定を読み込み
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


@app.get("/items/")
def run_model(text: str):
    #prompt = "<s>[INST] <<SYS>>あなたは有能なアシスタントです。 日本語で回答してください。<</SYS>>" + text + "<</INST>>"
    prompt = text 
    print(prompt)
    sequences = generation_pipe(
        prompt,
        max_length=128,    #生成する最大トークン数
        #pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        top_k=40,
        top_p=0.6  #単語のランダム性を指定
    )

    custom_headers = {
        "Access-Control-Allow-Origin": "*",
    }

    tmp = (sequences[0]["generated_text"])
    tmp = tmp.replace("\n", "")
    tmp = tmp.replace("\"", "")

    #return JSONResponse(content=(sequences[0]["generated_text"]).split("\n")[2], headers=custom_headers)
    return JSONResponse(content=tmp, headers=custom_headers)
