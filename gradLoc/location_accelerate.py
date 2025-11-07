from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import argparse
from typing import List
from accelerate import dispatch_model

# 模型路径
MODEL_PATH = "/VisCom-HDD-1/wyf/D3/llm/HuatuoGPT2-7B"

def build_chat_input(model, tokenizer, messages: List[dict], max_new_tokens: int = 0, return_type: str = "tensor"):
    """
    构造聊天输入，可输出：
      - return_type="tensor"：返回 token 张量，用于 model.generate()
      - return_type="text"：返回字符串，用于 pipeline()
    """
    def _parse_messages(messages, split_role="user"):
        system, rounds = "", []
        round = []
        for i, message in enumerate(messages):
            if message["role"] == split_role and round:
                rounds.append(round)
                round = []
            round.append(message)
        if round:
            rounds.append(round)
        return system, rounds

    max_new_tokens = max_new_tokens or getattr(model.generation_config, "max_new_tokens", 128)
    max_input_tokens = model.config.model_max_length - max_new_tokens
    system, rounds = _parse_messages(messages, split_role="user")
    roles = ('<问>：', '<答>：')
    sep = '\n'

    # ===============================
    # 构造历史内容（倒序累加，防止超长）
    # ===============================
    history_tokens = []
    for round in rounds[::-1]:
        round_tokens = []
        for message in round:
            if message["role"] == "user":
                round_tokens.extend(tokenizer.encode(roles[0] + message["content"] + sep))
            else:
                round_tokens.extend(tokenizer.encode(roles[1] + message["content"] + sep))
        if len(history_tokens) == 0 or len(history_tokens) + len(round_tokens) <= max_input_tokens:
            history_tokens = round_tokens + history_tokens
            if len(history_tokens) < max_input_tokens:
                continue
        break

    # ===============================
    # 补上答的起始标识
    # ===============================
    input_tokens = history_tokens
    if messages[-1]["role"] != "assistant":
        input_tokens.extend(tokenizer.encode(roles[1]))
    input_tokens = input_tokens[-max_input_tokens:]

    # ===============================
    # 返回不同类型
    # ===============================
    if return_type == "tensor":
        return torch.LongTensor([input_tokens]).to(model.device)
    elif return_type == "text":
        return tokenizer.decode(input_tokens, skip_special_tokens=True)
    else:
        raise ValueError("return_type must be either 'tensor' or 'text'")

def run_with_accelerate(user_input=None):
    print("=" * 40)
    print(" Running with Accelerate ".center(40, "="))
    print("=" * 40)

    # 1. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left", trust_remote_code=True)

    # 2. 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        local_files_only=True,
        trust_remote_code=True
    )

    # 3. 使用 Accelerate 分配模型（使用GPU 0）
    model = dispatch_model(model, device_map={"": 0})

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    messages = [{"role": "user", "content": user_input}]
    prompt = build_chat_input(model, tokenizer, messages, return_type="text")
    with torch.no_grad():
        output = pipe(
            prompt,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            max_new_tokens=2048,
            temperature=0.3,
            top_k=5,
            top_p=0.85,
            repetition_penalty=1.1,
            do_sample=True,
            return_full_text=False,
            use_cache=False
        )
    generated_text = output[0]["generated_text"].strip()

    print("Pipeline生成的新文字:")
    print(generated_text)
    print("生成结束")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_input", type=str, default="", help="用户输入的文本")
    args = parser.parse_args()
    run_with_accelerate(args.user_input)
