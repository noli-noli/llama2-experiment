from llama_cpp import Llama
from fastapi import FastAPI
from fastapi.responses import JSONResponse
#日本語入力用モジュール
import readline

app = FastAPI()

# LLMの準備
llm = Llama(model_path="../models/Llama-2-70B-chat-GGUF/llama-2-70b-chat.Q5_K_M.gguf",n_gpu_layers=83)

@app.get("/items/")
def run_model(text: str):
    prompt = text
    print("prompt: ", prompt)

    # 推論の実行
    output = llm(
        prompt,
        max_tokens=256,
        echo=False,
        top_p=0.6,
        top_k=40, 
    )

    custom_headers = {
        "Access-Control-Allow-Origin": "*",
    }
    print("output: ", (output["choices"][0]["text"]).replace("\n", ""))
    #return JSONResponse(content=(sequences[0]["generated_text"]).split("\n")[2], headers=custom_headers)
    return JSONResponse(content=(output["choices"][0]["text"]).replace("\n", ""), headers=custom_headers)
