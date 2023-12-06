from llama_cpp import Llama

#日本語入力用モジュール
import readline

# LLMの準備
llm = Llama(model_path="../models/Llama-2-70B-chat-GGUF/llama-2-70b.Q4_K_M.gguf",n_gpu_layers=83,n_ctx=512)
#llm = Llama(model_path="../models/Llama-2-70B-chat-GGUF/llama-2-70b-chat.Q5_K_M.gguf",n_gpu_layers=83,n_ctx=512)


while True:
    prompt = input("Please enter the prompt >> ")
    if prompt=="exit":
        break

    # 推論の実行
    output = llm(
        prompt,
        max_tokens=256,
        echo=False,
        top_p=0.85,
        top_k=10, 
    )

    print(output["choices"][0]["text"])
    