from transformers import AutoTokenizer
import torch
import argparse
from typing import List
from vllm import LLM, SamplingParams

# 模型路径
MODEL_PATH = "/VisCom-HDD-1/wyf/D3/llm/HuatuoGPT2-7B"

def build_chat_input(tokenizer, messages: List[dict], max_new_tokens: int = 0, return_type: str = "text"):
    """
    构造聊天输入，可输出：
      - return_type="text"：返回字符串，用于 vLLM
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

    system, rounds = _parse_messages(messages, split_role="user")
    roles = ('<问>：', '<答>：')
    sep = '\n'

    # ===============================
    # 构造历史内容
    # ===============================
    history_text = ""
    for round in rounds:
        for message in round:
            if message["role"] == "user":
                history_text += roles[0] + message["content"] + sep
            else:
                history_text += roles[1] + message["content"] + sep

    # ===============================
    # 补上答的起始标识
    # ===============================
    if messages[-1]["role"] != "assistant":
        history_text += roles[1]

    return history_text

def run_with_vllm(user_input=None):
    print("=" * 40)
    print(" Running with vLLM ".center(40, "="))
    print("=" * 40)

    # 1. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left", trust_remote_code=True)

    # 2. 初始化 vLLM 引擎
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        dtype="float16"
    )

    # 3. 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.3,
        top_k=5,
        top_p=0.85,
        repetition_penalty=1.1,
        max_tokens=2048
    )

    messages = [{"role": "user", "content": user_input}]
    prompt = build_chat_input(tokenizer, messages, return_type="text")

    # 4. 生成
    outputs = llm.generate([prompt], sampling_params)
    generated_text = outputs[0].outputs[0].text.strip()

    print("Pipeline生成的新文字:")
    print(generated_text)
    print("生成结束")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_input", type=str, default="", help="用户输入的文本")
    args = parser.parse_args()
    run_with_vllm(args.user_input)
