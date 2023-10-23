from llama_cpp import Llama

#日本語入力用モジュール
import readline

# LLMの準備
#llm = Llama(model_path="./Llama-2-70B-chat-GGUF/llama-2-70b-chat.Q5_K_M.gguf",n_gpu_layers=40)
llm = Llama(model_path="../models/Llama-2-70B-chat-GGUF/llama-2-70b-chat.Q5_K_M.gguf",n_gpu_layers=83)

while True:
    prompt = input("Please enter the prompt >> ")
    if prompt=="exit":
        break

    # 推論の実行
    output = llm(
        prompt,
        max_tokens=128,
        echo=False,
        top_p=0.6,
        top_k=40, 
    )

    print(output["choices"][0]["text"])