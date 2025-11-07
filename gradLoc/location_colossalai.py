import torch
import transformers
import colossalai
from colossalai.inference import InferenceEngine, InferenceConfig
import argparse
from typing import List

# 模型路径
MODEL_PATH = "/VisCom-HDD-1/wyf/D3/llm/Meta-Llama-3-8B"

def build_chat_input(tokenizer, messages: List[dict], return_type: str = "text"):
    """
    构造聊天输入
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

    # 构造历史内容
    history_text = ""
    for round in rounds:
        for message in round:
            if message["role"] == "user":
                history_text += roles[0] + message["content"] + sep
            else:
                history_text += roles[1] + message["content"] + sep

    # 补上答的起始标识
    if messages[-1]["role"] != "assistant":
        history_text += roles[1]

    return history_text

def run_with_colossalai(user_input=None):
    print("=" * 40)
    print(" Running with ColossalAI ".center(40, "="))
    print("=" * 40)

    # 启动 ColossalAI
    colossalai.launch_from_torch()

    # 1. 加载 tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True  # reduce CPU memory during loading
    )

    # 3. 创建推理配置
    inference_config = InferenceConfig(
        dtype=torch.float16,
        max_batch_size=1,
        max_input_len=1024,
        max_output_len=2048,
        use_cuda_kernel=False,
        tp_size=1
    )

    # 4. 创建推理引擎
    engine = InferenceEngine(model, tokenizer, inference_config, verbose=True)

    messages = [{"role": "user", "content": user_input}]
    prompt = build_chat_input(tokenizer, messages, return_type="text")

    # 5. 生成
    response = engine.generate(prompts=[prompt])

    if response and len(response) > 0:
        generated_text = response[0].strip()
    else:
        generated_text = ""

    print("Pipeline生成的新文字:")
    print(generated_text)
    print("生成结束")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_input", type=str, default="", help="用户输入的文本")
    args = parser.parse_args()
    run_with_colossalai(args.user_input)
