from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import transformers
import torch
import argparse
from typing import List
import json

# 设置日志级别以避免 generation_config 的 JSON 序列化错误
transformers.logging.set_verbosity_error()

# 模型路径
MODEL_PATH = "/VisCom-HDD-1/wyf/D3/llm/Meta-Llama-3-8B"

class OutputCollector:
    """收集模型中间层的输出"""

    def __init__(self):
        self.rotary_outputs = []
        self.norm_outputs = []
        self.hooks = []

    def clear(self):
        """清空收集的输出"""
        self.rotary_outputs.clear()
        self.norm_outputs.clear()

    def register_hooks(self, model):
        """在模型上注册hook以收集rotary_emb和norm的输出"""
        self.clear()
        self.remove_hooks()

        # 获取实际的模型
        actual_model = model.module if hasattr(model, 'module') else model

        # 遍历模型的所有模块，查找rotary_emb和norm
        for name, module in actual_model.named_modules():
            # 查找 Rotary Embedding 层
            if 'rotary_emb' in name.lower() or 'rope' in name.lower():
                hook = module.register_forward_hook(self._make_rotary_hook(name))
                self.hooks.append(hook)

            # 查找 LayerNorm 或 RMSNorm 层
            if 'norm' in name.lower():
                hook = module.register_forward_hook(self._make_norm_hook(name))
                self.hooks.append(hook)

    def _make_rotary_hook(self, layer_name: str):
        """创建rotary embedding的hook函数"""
        def hook(module, input, output):
            # 保存rotary embedding的输出的统计信息
            if isinstance(output, tuple):
                stats = []
                for o in output:
                    if torch.is_tensor(o):
                        stats.append({
                            'mean': float(o.mean().item()),
                            'std': float(o.std().item()),
                            'shape': list(o.shape)
                        })
                self.rotary_outputs.append({
                    'layer': layer_name,
                    'stats': stats
                })
            else:
                if torch.is_tensor(output):
                    self.rotary_outputs.append({
                        'layer': layer_name,
                        'stats': {
                            'mean': float(output.mean().item()),
                            'std': float(output.std().item()),
                            'shape': list(output.shape)
                        }
                    })
        return hook

    def _make_norm_hook(self, layer_name: str):
        """创建normalization的hook函数"""
        def hook(module, input, output):
            # 保存norm的输出的统计信息
            if torch.is_tensor(output):
                self.norm_outputs.append({
                    'layer': layer_name,
                    'stats': {
                        'mean': float(output.mean().item()),
                        'std': float(output.std().item()),
                        'shape': list(output.shape)
                    }
                })
        return hook

    def remove_hooks(self):
        """移除所有注册的hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def to_dict(self):
        """转换为可序列化的字典"""
        return {
            'rotary_outputs': self.rotary_outputs,
            'norm_outputs': self.norm_outputs
        }


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

def run_without_deepspeed(user_input=None):
    print("=" * 40)
    print(" Running without DeepSpeed ".center(40, "="))
    print("=" * 40)

    # 1. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left", trust_remote_code=True)

    # 2. 加载模型
    # Note: When CUDA_VISIBLE_DEVICES is set, the visible device index is 0
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        local_files_only=True,
        trust_remote_code=True
    ).to("cuda:0")

    # 3. 创建collector并注册hooks
    collector = OutputCollector()
    collector.register_hooks(model)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0
    )

    if 'HuatuoGPT2-7B' in MODEL_PATH:
        messages = [{"role": "user", "content": user_input}]
        prompt = build_chat_input(model, tokenizer, messages, return_type="text")
    elif 'vicuna' in MODEL_PATH:
        prompt = "A chat between a user and an assistant.\n"
        prompt += f"USER: {user_input}\nASSISTANT:"
    elif 'Llama' in MODEL_PATH:
        prompt = user_input
    else:
        raise ValueError("Unknown model type in MODEL_PATH")

    print("输入的prompt:")
    print(prompt)

    with torch.no_grad():
        if 'HuatuoGPT2-7B' in MODEL_PATH:
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
        elif 'vicuna' in MODEL_PATH:
            output = pipe(
                prompt,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
                max_new_tokens=2048,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                return_full_text=False,
                use_cache=True
            )
        elif 'Llama' in MODEL_PATH:
            output = pipe(
                prompt,
                bos_token_id=128000,
                eos_token_id=128001,
                max_new_tokens=2048,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                return_full_text=False,
                use_cache=True
            )
        else:
            raise ValueError("Unknown model type in MODEL_PATH")

    generated_text = output[0]["generated_text"].strip()

    print("Pipeline生成的新文字:")
    print(generated_text)
    print("生成结束")

    # 输出中间层数据（JSON格式）
    print("INTERMEDIATE_OUTPUTS_START")
    print(json.dumps(collector.to_dict()))
    print("INTERMEDIATE_OUTPUTS_END")

    collector.remove_hooks()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--user_input",
        type=str,
        default="【格式要求：必须遵守，否则回答无效】\n1. 第一行：仅写大写字母选项（A/B/C/D/E其一），无任何多余文字、标点、空格；\n2. 第二行起：详细解释（精神神经系统科室场景，中文说明选择依据，需分析每个选项）。\n\n问题：中颅凹骨折的典型表现是？\n选项：\nA. 鼻流血\nB. 双眼睑皮下青紫，逐渐加重\nC. 乳突下或咽后壁黏膜下淤血\nD. 脑脊液耳漏\nE. 颞部头皮肿胀淤血",
        help="用户输入的文本（包含问题、选项和格式要求）"
    )
    args = parser.parse_args()
    run_without_deepspeed(args.user_input)
