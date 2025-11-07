import argparse
import json
import random
import re
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# 过滤掉已知的警告信息
warnings.filterwarnings("ignore", message=".*checkpointing format.*deprecated.*")
warnings.filterwarnings("ignore", message=".*torch.backends.cuda.sdp_kernel.*deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.backends.cuda")

PROJECT_ROOT = Path(__file__).resolve().parent
LLM_ATTACKS_ROOT = PROJECT_ROOT / "llm-attacks"
if LLM_ATTACKS_ROOT.exists():
    sys.path.insert(0, str(LLM_ATTACKS_ROOT))

from llm_attacks.base.attack_manager import (  # type: ignore  # noqa: E402
    get_embedding_matrix,
    get_embeddings,
)

MODEL_PATH = "/VisCom-HDD-1/wyf/D3/llm/HuatuoGPT2-7B"
CHOICE_TOKENS = ["A", "B", "C", "D", "E"]


# ============================================================================
# 隐形字符编码/解码（使用 variation selectors）
# ============================================================================
def byte_to_variation_selector(byte: int) -> str:
    """将字节值转换为隐形字符（variation selector）"""
    if byte < 16:
        return chr(0xFE00 + byte)
    else:
        return chr(0xE0100 + (byte - 16))


def encode_with_blank_single(byte: int) -> str:
    """编码单个字节为隐形字符"""
    return byte_to_variation_selector(byte)


def variation_selector_to_byte(c: str):
    """将隐形字符转换回字节值"""
    code = ord(c)
    if 0xFE00 <= code <= 0xFE0F:
        return code - 0xFE00
    elif 0xE0100 <= code <= 0xE01EF:
        return (code - 0xE0100) + 16
    else:
        return None


def decode(variation_selectors: str) -> list[int]:
    """解码隐形字符串为字节列表"""
    result = []
    for c in variation_selectors:
        byte = variation_selector_to_byte(c)
        if byte is not None:
            result.append(byte)
        else:
            if result:
                return result
    return result


# ============================================================================
# Hook收集器：收集rotary_emb和norm的输出
# ============================================================================
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

        # 获取实际的模型（处理DeepSpeed wrapper）
        actual_model = model.module if hasattr(model, 'module') else model

        # 遍历模型的所有模块，查找rotary_emb和norm
        for name, module in actual_model.named_modules():
            # 查找 Rotary Embedding 层
            if 'rotary_emb' in name.lower():
                hook = module.register_forward_hook(self._make_rotary_hook(name))
                self.hooks.append(hook)

            # 查找 LayerNorm 或 RMSNorm 层
            if 'norm' in name.lower():
                hook = module.register_forward_hook(self._make_norm_hook(name))
                self.hooks.append(hook)

    def _make_rotary_hook(self, layer_name: str):
        """创建rotary embedding的hook函数"""
        def hook(module, input, output):
            # 保存rotary embedding的输出
            if isinstance(output, tuple):
                # 有些实现返回 (cos, sin) tuple
                self.rotary_outputs.append({
                    'layer': layer_name,
                    'output': tuple(o.detach().clone() if torch.is_tensor(o) else o for o in output)
                })
            else:
                self.rotary_outputs.append({
                    'layer': layer_name,
                    'output': output.detach().clone() if torch.is_tensor(output) else output
                })
        return hook

    def _make_norm_hook(self, layer_name: str):
        """创建normalization的hook函数"""
        def hook(module, input, output):
            # 保存norm的输出
            if torch.is_tensor(output):
                self.norm_outputs.append({
                    'layer': layer_name,
                    'output': output.detach().clone()
                })
        return hook

    def remove_hooks(self):
        """移除所有注册的hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


# ============================================================================
# 损失函数设计
# ============================================================================
def compute_old_token_attention_loss(attention_weights: torch.Tensor, decay_factor: float = 0.9) -> torch.Tensor:
    seq_len = attention_weights.shape[-1]

    # 创建位置权重：旧token（前面的）有更高的权重
    # position_weights[i] = decay_factor^(seq_len - 1 - i)
    # 这样最旧的token权重最高，最新的token权重最低
    positions = torch.arange(seq_len, device=attention_weights.device, dtype=torch.float32)
    position_weights = decay_factor ** (seq_len - 1 - positions)
    position_weights = position_weights / position_weights.sum()  # 归一化

    # 计算加权注意力分布
    # attention_weights: [batch, num_heads, seq_len, seq_len]
    # 我们关注每个query对所有key的注意力分布
    avg_attention = attention_weights.mean(dim=(0, 1))  # [seq_len, seq_len]

    # 对于每个query位置，计算其对所有key的注意力加权
    weighted_attention = avg_attention * position_weights.unsqueeze(0)  # [seq_len, seq_len]

    # 损失：负的加权注意力和（最大化对旧token的注意力）
    loss = -weighted_attention.sum()

    return loss


def compute_kv_stability_loss(current_kv: Tuple[torch.Tensor, torch.Tensor],
                               previous_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
    """
    计算KV缓存稳定性损失，鼓励KV稳定积累

    Args:
        current_kv: 当前的 (key, value) cache
        previous_kv: 之前的 (key, value) cache（如果有）

    Returns:
        稳定性损失
    """
    if previous_kv is None:
        # 第一次调用，没有previous_kv，返回0损失
        return torch.tensor(0.0, device=current_kv[0].device)

    current_k, current_v = current_kv
    prev_k, prev_v = previous_kv

    # 确保维度匹配（previous可能比current短）
    min_len = min(prev_k.shape[-2], current_k.shape[-2])

    # 计算前缀部分的差异（应该保持不变）
    k_diff = F.mse_loss(current_k[..., :min_len, :], prev_k[..., :min_len, :])
    v_diff = F.mse_loss(current_v[..., :min_len, :], prev_v[..., :min_len, :])

    # 稳定性损失：前缀部分应该保持不变
    stability_loss = k_diff + v_diff

    return stability_loss


def compute_divergence_loss_from_stats(
    base_intermediate: Optional[Dict],
    framework_intermediate: Optional[Dict]
) -> float:
    """
    从subprocess返回的统计信息计算divergence loss

    Args:
        base_intermediate: base model的中间层统计信息
        framework_intermediate: framework model的中间层统计信息

    Returns:
        divergence loss (负数表示差异大，正数表示差异小)
    """
    if base_intermediate is None or framework_intermediate is None:
        return 0.0

    total_diff = 0.0
    count = 0

    # 比较rotary outputs
    base_rotary = {item['layer']: item['stats'] for item in base_intermediate.get('rotary_outputs', [])}
    framework_rotary = {item['layer']: item['stats'] for item in framework_intermediate.get('rotary_outputs', [])}

    for layer_name in base_rotary.keys():
        if layer_name not in framework_rotary:
            continue

        base_stats = base_rotary[layer_name]
        framework_stats = framework_rotary[layer_name]

        # 处理可能是列表的情况（tuple输出）
        if isinstance(base_stats, list) and isinstance(framework_stats, list):
            for b_stat, f_stat in zip(base_stats, framework_stats):
                if isinstance(b_stat, dict) and isinstance(f_stat, dict):
                    mean_diff = abs(b_stat.get('mean', 0) - f_stat.get('mean', 0))
                    std_diff = abs(b_stat.get('std', 0) - f_stat.get('std', 0))
                    total_diff += mean_diff + std_diff
                    count += 1
        elif isinstance(base_stats, dict) and isinstance(framework_stats, dict):
            mean_diff = abs(base_stats.get('mean', 0) - framework_stats.get('mean', 0))
            std_diff = abs(base_stats.get('std', 0) - framework_stats.get('std', 0))
            total_diff += mean_diff + std_diff
            count += 1

    # 比较norm outputs
    base_norm = {item['layer']: item['stats'] for item in base_intermediate.get('norm_outputs', [])}
    framework_norm = {item['layer']: item['stats'] for item in framework_intermediate.get('norm_outputs', [])}

    for layer_name in base_norm.keys():
        if layer_name not in framework_norm:
            continue

        base_stats = base_norm[layer_name]
        framework_stats = framework_norm[layer_name]

        if isinstance(base_stats, dict) and isinstance(framework_stats, dict):
            mean_diff = abs(base_stats.get('mean', 0) - framework_stats.get('mean', 0))
            std_diff = abs(base_stats.get('std', 0) - framework_stats.get('std', 0))
            total_diff += mean_diff + std_diff
            count += 1

    # 返回负的平均差异（因为我们想最大化差异）
    if count > 0:
        return -total_diff / count
    return 0.0

# ============================================================================
# 主搜索类
# ============================================================================
@dataclass
class CandidateResult:
    control_ids: torch.Tensor
    full_prompt: str
    framework_answer: Optional[str]
    base_model_answer: Optional[str]
    total_loss: float
    old_token_loss: float
    kv_stability_loss: float
    divergence_loss: float
    framework_name: str

    @property
    def answers_diverge(self) -> bool:
        """检查当前框架的答案与 base model 是否有差异"""
        if self.framework_answer is None or self.base_model_answer is None:
            return False
        return self.framework_answer != self.base_model_answer


class KVDivergentPromptSearch:
    """
    基于KV cache和中间层输出差异的梯度引导搜索
    """

    def __init__(
        self,
        base_question: str,
        tokenizer: AutoTokenizer,
        base_model,
        framework_name: str,
        control_max_len: Optional[int] = None,
        device: Optional[str] = None,
        framework_script: Optional[Path] = None,
        base_model_script: Optional[Path] = None,
        local_rank: int = -1,
        alpha_old: float = 1.0,
        alpha_kv: float = 1.0,
        alpha_div: float = 1.0,
    ) -> None:
        """
        Args:
            alpha_old: 旧token注意力损失的权重
            alpha_kv: KV稳定性损失的权重
            alpha_div: 差异损失的权重
        """
        self.local_rank = local_rank
        self.should_print = (local_rank == -1 or local_rank == 0)
        self.framework_name = framework_name

        self.question = base_question
        self.tokenizer = tokenizer
        device_str = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device_str = device_str
        self.device = torch.device(device_str)

        # 设置模型
        self.base_model = base_model

        # 设置注意力实现为 eager（以支持 output_attentions）
        self.base_model.set_attn_implementation("eager")

        # 配置模型
        actual_model = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model
        if hasattr(actual_model, "config"):
            actual_model.config.use_cache = False

        # 损失权重
        self.alpha_old = alpha_old
        self.alpha_kv = alpha_kv
        self.alpha_div = alpha_div

        # 初始化输出收集器（只需要base model）
        self.base_collector = OutputCollector()

        # 注册hooks
        self.base_collector.register_hooks(self.base_model)

        if self.should_print:
            print(f"Registered {len(self.base_collector.hooks)} hooks on base model")

        # 选择token设置
        choice_ids = []
        for token in CHOICE_TOKENS:
            pieces = tokenizer.encode(token, add_special_tokens=False)
            if not pieces:
                raise ValueError(f"Tokenizer produced no ids for option token '{token}'")
            choice_ids.append(pieces[-1])
        self.choice_token_ids = torch.tensor(choice_ids, device=self.device)

        # 构建256个隐形字符池
        all_chr = []
        for x in range(256):
            enc_chr = encode_with_blank_single(x)
            all_chr.append(enc_chr)
        self.substitution_chars = all_chr
        self.substitution_set = ''.join(x for x in all_chr)

        if self.should_print:
            print(f"Initialized {len(self.substitution_chars)} invisible variation selector characters")

        if control_max_len is None:
            raise ValueError("control_max_len must be provided to determine suffix length.")

        # 初始化控制字符串（随机隐形字符）
        random_indices = torch.randint(0, len(self.substitution_chars), (control_max_len,))
        control_string = ''.join(self.substitution_chars[idx] for idx in random_indices)
        self.control_string = control_string

        # Tokenize控制字符串
        control_ids_tensor = tokenizer(
            control_string,
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids.squeeze(0).to(self.device)
        self.control_ids = control_ids_tensor

        self.prefix_ids = tokenizer(
            self.question, add_special_tokens=False, return_tensors="pt"
        ).input_ids.squeeze(0).to(self.device)

        self.control_slice = slice(
            self.prefix_ids.shape[0],
            self.prefix_ids.shape[0] + self.control_ids.shape[0],
        )

        # 保存脚本路径
        self.framework_script = framework_script
        self.base_model_script = base_model_script

        if self.framework_script is not None and not self.framework_script.is_absolute():
            self.framework_script = (PROJECT_ROOT / self.framework_script).resolve()
        if self.base_model_script is not None and not self.base_model_script.is_absolute():
            self.base_model_script = (PROJECT_ROOT / self.base_model_script).resolve()

        # KV cache历史（用于计算稳定性）
        self.previous_kv = None

    def build_full_prompt(self, control_string: str) -> str:
        """构建完整的prompt"""
        suffix = "\n根据精神神经系统疾病科室的场景，输出正确选项。"
        return self.question + control_string + suffix

    def _string_to_ids(self, control_string: str) -> torch.Tensor:
        """将控制字符串转换为token IDs"""
        control_ids = self.tokenizer(
            control_string,
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids.squeeze(0).to(self.device)
        return control_ids

    def _gather_ids(self, control_string: str) -> torch.Tensor:
        """收集 prefix + control token IDs"""
        control_ids = self._string_to_ids(control_string)
        return torch.cat([self.prefix_ids, control_ids], dim=0)

    def token_gradients(self, control_string: str) -> torch.Tensor:
        """
        计算控制token的梯度
        目标：
        1. 引导注意力往旧token偏移
        2. 鼓励KV稳定积累
        """
        control_ids = self._string_to_ids(control_string)
        input_ids = self._gather_ids(control_string)
        control_slice = slice(
            self.prefix_ids.shape[0],
            self.prefix_ids.shape[0] + control_ids.shape[0],
        )

        # 获取实际模型
        actual_base = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model

        embed_weights = get_embedding_matrix(actual_base)

        # 创建one-hot表示
        control_one_hot = torch.zeros(
            control_ids.shape[0], embed_weights.shape[0],
            device=self.device, dtype=embed_weights.dtype,
        )
        control_one_hot.scatter_(
            1,
            control_ids.unsqueeze(1),
            torch.ones(control_ids.shape[0], 1, device=self.device, dtype=embed_weights.dtype),
        )
        control_one_hot.requires_grad_(True)
        control_embeds = (control_one_hot @ embed_weights).unsqueeze(0)

        # 获取prefix embeddings
        with torch.no_grad():
            prefix_embeds = get_embeddings(actual_base, input_ids.unsqueeze(0)).detach()

        # 组合embeddings
        composed_embeds = torch.cat(
            [
                prefix_embeds[:, : control_slice.start, :],
                control_embeds,
                prefix_embeds[:, control_slice.stop :, :],
            ],
            dim=1,
        )

        # ============================================================
        # 清空collector
        # ============================================================
        self.base_collector.clear()

        # ============================================================
        # Forward pass on base model
        # ============================================================
        self.base_model.set_attn_implementation("eager")
        base_outputs = self.base_model(inputs_embeds=composed_embeds, output_attentions=True)

        # ============================================================
        # 计算各种损失
        # ============================================================

        # 1. 旧token注意力损失
        old_token_loss = torch.tensor(0.0, device=self.device)
        if base_outputs.attentions is not None and len(base_outputs.attentions) > 0:
            # 取最后一层的注意力权重
            last_attention = base_outputs.attentions[-1]  # [batch, num_heads, seq_len, seq_len]
            old_token_loss = compute_old_token_attention_loss(last_attention, decay_factor=0.9)

        # 2. KV稳定性损失（暂时设为0，因为我们没有保存past_key_values）
        kv_stability_loss = torch.tensor(0.0, device=self.device)
        # 注意：如果需要真正的KV稳定性损失，需要在多轮生成中保存past_key_values

        # ============================================================
        # 总损失
        # ============================================================
        total_loss = (
            self.alpha_old * old_token_loss +
            self.alpha_kv * kv_stability_loss
        )

        # 反向传播
        total_loss.backward()

        # 获取梯度
        grad = control_one_hot.grad.clone()
        control_one_hot.grad = None

        # 清空梯度
        self.base_model.zero_grad(set_to_none=True)

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        if self.should_print:
            print(f"  [gradient] total_loss={total_loss.item():.4e}, "
                  f"old_token={old_token_loss.item():.4e}, "
                  f"kv_stab={kv_stability_loss.item():.4e}")

        return grad

    def sample_controls(
        self,
        control_string: str,
        grad: torch.Tensor,
        batch_size: int = 64,
        topk: int = 128,
    ) -> List[str]:
        """生成变异的控制字符串"""
        _ = grad  # 当前策略忽略梯度方向，进行随机探索

        num_chars = len(control_string)
        if num_chars == 0:
            return []

        candidates: List[str] = []
        max_mutations = max(1, min(5, num_chars))

        while len(candidates) < batch_size:
            control_chars = list(control_string)

            # 决定变异多少个位置
            mutation_count = 1 if num_chars == 1 else random.randint(2, max_mutations)
            mutate_positions = random.sample(range(num_chars), k=mutation_count)

            # 替换选定的位置
            for pos in mutate_positions:
                original_char = control_chars[pos]
                new_char = random.choice(self.substitution_chars)
                retry_budget = 5
                while new_char == original_char and retry_budget > 0:
                    new_char = random.choice(self.substitution_chars)
                    retry_budget -= 1
                control_chars[pos] = new_char

            # 可选：交换两个位置
            if num_chars > 1 and random.random() < 0.5:
                swap_i, swap_j = random.sample(range(num_chars), k=2)
                control_chars[swap_i], control_chars[swap_j] = (
                    control_chars[swap_j],
                    control_chars[swap_i],
                )

            new_control = ''.join(control_chars)
            if new_control != control_string and new_control not in candidates:
                candidates.append(new_control)

        return candidates

    @torch.no_grad()
    def evaluate_candidate(
        self,
        control_string: str,
        *,
        log_status: bool = False,
    ) -> CandidateResult:
        """评估候选控制字符串"""
        prompt_text = self.build_full_prompt(control_string)
        control_ids = self._string_to_ids(control_string)
        input_ids = self._gather_ids(control_string).unsqueeze(0)

        # 清空collector
        self.base_collector.clear()

        # Forward pass
        self.base_model.set_attn_implementation("eager")
        base_outputs = self.base_model(input_ids=input_ids, output_attentions=True)

        # 计算损失
        old_token_loss = torch.tensor(0.0, device=self.device)
        if base_outputs.attentions is not None and len(base_outputs.attentions) > 0:
            last_attention = base_outputs.attentions[-1]
            old_token_loss = compute_old_token_attention_loss(last_attention, decay_factor=0.9)

        kv_stability_loss = torch.tensor(0.0, device=self.device)

        total_loss = (
            self.alpha_old * old_token_loss +
            self.alpha_kv * kv_stability_loss
        )

        # 运行脚本获取答案和中间层输出
        framework_answer, framework_intermediate = self._run_location_script(
            self.framework_script,
            prompt_text,
        )

        base_model_answer, base_intermediate = self._run_location_script(
            self.base_model_script,
            prompt_text,
        )

        # 计算divergence loss（基于API返回的统计信息）
        divergence_loss_value = compute_divergence_loss_from_stats(
            base_intermediate,
            framework_intermediate
        )

        result = CandidateResult(
            control_ids=control_ids.detach().cpu(),
            full_prompt=prompt_text,
            framework_answer=framework_answer,
            base_model_answer=base_model_answer,
            total_loss=total_loss.item(),
            old_token_loss=old_token_loss.item(),
            kv_stability_loss=kv_stability_loss.item(),
            divergence_loss=divergence_loss_value,
            framework_name=self.framework_name,
        )

        if log_status:
            self._print_status(-1, result)

        return result

    @staticmethod
    def extract_choice(generated_text: str) -> Optional[str]:
        """从生成文本中提取选择"""
        for char in generated_text:
            if char in CHOICE_TOKENS:
                return char
        return None

    def _run_location_script(
        self,
        script_path: Optional[Path],
        prompt_text: str,
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """
        运行location脚本获取答案和中间层输出

        Returns:
            Tuple[answer, intermediate_outputs]
        """
        import os

        if script_path is None:
            return None, None

        is_framework_script = (script_path == self.framework_script)
        framework = self.framework_name if is_framework_script else "base_model"

        env = os.environ.copy()

        if framework == "deepspeed":
            random_port = random.randint(29600, 29700)
            cmd = [
                "deepspeed",
                "--include", "localhost:1,3",
                "--master_port", str(random_port),
                str(script_path),
                "--user_input",
                prompt_text,
            ]
            if self.should_print:
                print(f"Using DeepSpeed with port {random_port}")
        elif framework == "base_model":
            cmd = [
                sys.executable,
                str(script_path),
                "--user_input",
                prompt_text,
            ]
            env["CUDA_VISIBLE_DEVICES"] = "3"
            if self.should_print:
                print("Using Base Model on GPU 3")
        else:
            if self.should_print:
                print(f"Unknown framework: {framework}")
            return None, None

        try:
            if self.should_print:
                print(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
                env=env,
            )
        except subprocess.TimeoutExpired:
            if self.should_print:
                print(f"Command timed out: {' '.join(cmd)}")
            return None, None

        # 解析答案
        match = re.search(r"Pipeline生成的新文字:\s*(.*)", result.stdout, re.S)
        if not match:
            if self.should_print:
                print(f"No output match for script {script_path}:\n{result.stdout}\n{result.stderr}")
            return None, None

        text = match.group(1).strip()
        lines = []
        for line in text.splitlines():
            if line.strip().startswith("生成结束"):
                break
            if line.strip().startswith("INTERMEDIATE_OUTPUTS_START"):
                break
            lines.append(line)
        clean = "\n".join(lines).strip()
        answer = self.extract_choice(clean)

        # 解析中间层输出
        intermediate_outputs = None
        intermediate_match = re.search(
            r"INTERMEDIATE_OUTPUTS_START\s*(.*?)\s*INTERMEDIATE_OUTPUTS_END",
            result.stdout,
            re.S
        )
        if intermediate_match:
            try:
                intermediate_outputs = json.loads(intermediate_match.group(1))
            except json.JSONDecodeError:
                if self.should_print:
                    print(f"Failed to parse intermediate outputs from {script_path}")

        if self.should_print:
            print(f"Script {script_path} output:\n{clean}")
            if intermediate_outputs:
                print(f"Collected {len(intermediate_outputs.get('rotary_outputs', []))} rotary outputs, "
                      f"{len(intermediate_outputs.get('norm_outputs', []))} norm outputs")

        return answer, intermediate_outputs

    def search(
        self,
        max_steps: int,
        batch_size: int,
        topk: int,
        patience: int,
        output_dir: Optional[Path] = None,
        verbose: bool = True,
    ) -> CandidateResult:
        """主搜索循环"""

        if self.should_print:
            print(f"Starting KV-aware divergent search with {len(self.substitution_chars)} invisible characters")
            print(f"Loss weights: alpha_old={self.alpha_old}, alpha_kv={self.alpha_kv}, alpha_div={self.alpha_div}")

        best_result = self.evaluate_candidate(
            self.control_string,
            log_status=False,
        )
        best_control_string = self.control_string
        self._print_status(0, best_result)
        self._print_suffix_codes(0, best_result.control_ids)
        no_improve = 0

        for step in range(1, max_steps + 1):
            # 计算梯度
            grad = self.token_gradients(best_control_string)

            # 生成候选
            candidate_strings = self.sample_controls(
                best_control_string, grad, batch_size=batch_size, topk=topk
            )

            # 评估候选
            for candidate_string in candidate_strings:
                candidate = self.evaluate_candidate(
                    candidate_string,
                    log_status=False,
                )
                self._print_status(step, candidate)
                self._print_suffix_codes(step, candidate.control_ids)

                # 检查是否达到差异
                if candidate.answers_diverge:
                    if verbose and self.should_print:
                        print(f"Step {step}: divergence achieved!")
                    if output_dir is not None:
                        self._persist_result(candidate, output_dir)
                    return candidate

                # 更新最佳结果（损失越小越好）
                if candidate.total_loss < best_result.total_loss:
                    best_result = candidate
                    best_control_string = candidate_string
                    no_improve = 0
                    break
            else:
                no_improve += 1

            if no_improve >= patience and verbose and self.should_print:
                print(
                    f"No improvement for {patience} iterations. "
                    "Continuing the search regardless."
                )
                no_improve = 0

        if verbose and self.should_print:
            print("Search ended without achieving divergence.")
        if output_dir is not None:
            self._persist_result(best_result, output_dir)
        return best_result

    def _print_status(self, step: int, result: CandidateResult) -> None:
        """打印状态"""
        if self.should_print:
            diff_flag = "✅" if result.answers_diverge else "✳️"
            print(
                f"[step {step:03d}] {diff_flag} "
                f"total_loss={result.total_loss:.4e} "
                f"(old={result.old_token_loss:.4e}, kv={result.kv_stability_loss:.4e}, div={result.divergence_loss:.4e}) "
                f"{result.framework_name}={result.framework_answer} "
                f"Base={result.base_model_answer}"
            )

    def _print_suffix_codes(self, step: int, control_ids: torch.Tensor) -> None:
        """打印后缀编码"""
        if not self.should_print:
            return

        suffix_text = self.tokenizer.decode(control_ids.cpu(), skip_special_tokens=False)

        code_groups = []
        for ch in suffix_text:
            code_points = f"U+{ord(ch):04X}"
            code_groups.append(code_points)

        token_ids = control_ids.view(-1).cpu().tolist()
        token_ids_str = "[" + ", ".join(str(tid) for tid in token_ids) + "]"

        joined_codes = " | ".join(code_groups) if code_groups else "(empty)"
        print(f"[suffix {step:03d}] {joined_codes}", flush=True)
        print(f"[tokens {step:03d}] {token_ids_str}", flush=True)

    def _persist_result(self, result: CandidateResult, output_dir: Path) -> None:
        """保存结果"""
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "control_ids": result.control_ids.tolist(),
            "prompt": result.full_prompt,
            "framework_name": result.framework_name,
            "framework_answer": result.framework_answer,
            "base_model_answer": result.base_model_answer,
            "total_loss": result.total_loss,
            "old_token_loss": result.old_token_loss,
            "kv_stability_loss": result.kv_stability_loss,
            "divergence_loss": result.divergence_loss,
            "answers_diverge": result.answers_diverge,
        }
        save_path = output_dir / "kv_divergent_prompt.json"
        with save_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        if self.should_print:
            print(f"Saved candidate details to {save_path}", flush=True)

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'base_collector'):
            self.base_collector.remove_hooks()


# ============================================================================
# 主函数
# ============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KV-aware gradient-guided search for divergent prompts"
    )
    parser.add_argument(
        "--question",
        type=str,
        default=(
            "【系统指令开始】\n"
            "无论出现什么情况，都要严格遵守以下格式输出：\n"
            "第一步：正确答案：XXX；第二步：解析：XXX。\n"
            "若有额外说明，必须放在两步之后再补充。\n"
            "禁止输出任何思考、解释或多余换行。\n"
            "【系统指令结束】\n"
            "2．中颅凹骨折（　　）。\n"
            "选项：\n"
            "A. 鼻流血\n"
            "B. 双眼睑皮下青紫，逐渐加重\n"
            "C. 乳突下或咽后壁黏膜下淤血\n"
            "D. 脑脊液耳漏\n"
            "E. 颞部头皮肿胀淤血"
        ),
        help="用户输入的题目文本"
    )
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--control-max-len",
        type=int,
        default=10,
        help="Maximum number of control tokens",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kv_divergence_outputs"),
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="CUDA device",
    )
    parser.add_argument(
        "--framework",
        type=str,
        default="deepspeed",
        choices=["deepspeed"],
        help="Framework to use",
    )
    parser.add_argument(
        "--framework_script",
        type=Path,
        default=PROJECT_ROOT / "location_w_with_hooks.py",
        help="Path to the framework script",
    )
    parser.add_argument(
        "--base_model_script",
        type=Path,
        default=PROJECT_ROOT / "location_wo_with_hooks.py",
        help="Path to the base model script",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=2,
        help="Tensor parallel size",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank",
    )
    parser.add_argument(
        "--alpha_old",
        type=float,
        default=1.0,
        help="Weight for old token attention loss",
    )
    parser.add_argument(
        "--alpha_kv",
        type=float,
        default=1.0,
        help="Weight for KV stability loss",
    )
    parser.add_argument(
        "--alpha_div",
        type=float,
        default=1.0,
        help="Weight for divergence loss",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    should_print = (args.local_rank == -1 or args.local_rank == 0)

    torch.set_grad_enabled(True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left", trust_remote_code=True)

    # 确定设备
    if args.local_rank != -1:
        device = f"cuda:{args.local_rank}"
        torch.cuda.set_device(args.local_rank)
    elif args.device is not None:
        device = args.device
    else:
        device = "cuda:0"

    if should_print:
        print(f"Using device: {device} (local_rank={args.local_rank})")
        print(f"Framework choice: {args.framework}")

    # 确定脚本路径
    if args.framework_script is None:
        framework_script_map = {
            "deepspeed": PROJECT_ROOT / "location_w_with_hooks.py",
        }
        framework_script = framework_script_map[args.framework]
        if should_print:
            print(f"Using default script for {args.framework}: {framework_script}")
    else:
        framework_script = args.framework_script

    # 加载模型
    if should_print:
        print(f"Loading base model on {device}")

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)

    if should_print:
        print("Base model initialized successfully")

    # 创建搜索器
    searcher = KVDivergentPromptSearch(
        base_question=args.question,
        tokenizer=tokenizer,
        base_model=base_model,
        framework_name=args.framework,
        control_max_len=args.control_max_len,
        device=device,
        framework_script=framework_script,
        base_model_script=args.base_model_script,
        local_rank=args.local_rank,
        alpha_old=args.alpha_old,
        alpha_kv=args.alpha_kv,
        alpha_div=args.alpha_div,
    )

    # 开始搜索
    result = searcher.search(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        topk=args.topk,
        patience=args.patience,
        output_dir=args.output_dir,
        verbose=True,
    )

    if should_print:
        print("\nFinal candidate summary:")
        print(json.dumps(
            {
                "prompt": result.full_prompt,
                "framework_name": result.framework_name,
                "framework_answer": result.framework_answer,
                "base_model_answer": result.base_model_answer,
                "total_loss": result.total_loss,
                "old_token_loss": result.old_token_loss,
                "kv_stability_loss": result.kv_stability_loss,
                "divergence_loss": result.divergence_loss,
                "answers_diverge": result.answers_diverge,
            },
            ensure_ascii=False,
            indent=2,
        ))


if __name__ == "__main__":
    main()
