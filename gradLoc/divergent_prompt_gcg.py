import argparse
import json
import random
import re
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
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

MODEL_PATH = "/VisCom-HDD-1/wyf/D3/llm/Meta-Llama-3-8B"
CHOICE_TOKENS = ["A", "B", "C", "D", "E"]

def byte_to_variation_selector(byte: int) -> str:
    if byte < 16:
        return chr(0xFE00 + byte)
    else:
        return chr(0xE0100 + (byte - 16))


def encode_with_blank_single(byte: int) -> str:
    result = byte_to_variation_selector(byte)
    return result


def variation_selector_to_byte(c: str):
    code = ord(c)
    if 0xFE00 <= code <= 0xFE0F:
        return code - 0xFE00
    elif 0xE0100 <= code <= 0xE01EF:
        return (code - 0xE0100) + 16
    else:
        return None


def decode(variation_selectors: str) -> list[int]:
    result = []
    for c in variation_selectors:
        byte = variation_selector_to_byte(c)
        if byte is not None:
            result.append(byte)
        else:
            if result:
                return result
    return result

@dataclass
class CandidateResult:
    control_ids: torch.Tensor
    full_prompt: str
    framework_answer: Optional[str]  # 当前框架的答案
    base_model_answer: Optional[str]  # base model 的答案（用于对比）
    uniform_loss: float
    framework_name: str  # 框架名称

    @property
    def answers_diverge(self) -> bool:
        """检查当前框架的答案与 base model 是否有差异"""
        if self.framework_answer is None or self.base_model_answer is None:
            return False
        return self.framework_answer != self.base_model_answer


class DivergentPromptSearch:
    """Gradient-guided search for divergent multiple-choice outputs."""

    def __init__(
        self,
        base_question: str,
        tokenizer: AutoTokenizer,
        base_model,  # Can be AutoModelForCausalLM or framework-specific engine
        framework_name: str,  # 当前使用的框架名称：deepspeed, vllm, accelerate, colossalai
        control_max_len: Optional[int] = None,
        device: Optional[str] = None,
        framework_script: Optional[Path] = None,  # 当前框架的脚本路径
        base_model_script: Optional[Path] = None,
        local_rank: int = -1,
        framework_model=None,  # 当前框架的模型（用于梯度计算）
    ) -> None:
        self.local_rank = local_rank
        self.should_print = (local_rank == -1 or local_rank == 0)
        self.framework_name = framework_name  # 保存框架名称

        if framework_name not in ["deepspeed", "vllm", "accelerate", "colossalai"]:
            raise ValueError(f"Invalid framework name: {framework_name}. Must be one of: deepspeed, vllm, accelerate, colossalai")

        self.question = base_question
        self.tokenizer = tokenizer
        device_str = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device_str = device_str
        self.device = torch.device(device_str)

        # Check if base_model is a framework engine (e.g., DeepSpeed)
        if hasattr(base_model, 'module'):
            # Framework engine - use it directly without moving to device
            if self.should_print:
                print(f"Detected framework engine for base_model: {framework_name}")
            self.model = base_model
            # Access the underlying module for config
            if hasattr(self.model.module, "config"):
                self.model.module.config.use_cache = False
        else:
            # Regular PyTorch model - move to device
            self.model = base_model.to(self.device)
            if hasattr(self.model, "config"):
                self.model.config.use_cache = False
            if hasattr(self.model, "gradient_checkpointing_enable"):
                try:
                    self.model.gradient_checkpointing_enable()
                except Exception as e:
                    if self.should_print:
                        print(f"Warning: Could not enable gradient checkpointing: {e}")

        self.model.eval()

        choice_ids = []
        for token in CHOICE_TOKENS:
            pieces = tokenizer.encode(token, add_special_tokens=False)
            if not pieces:
                raise ValueError(f"Tokenizer produced no ids for option token '{token}'")
            choice_ids.append(pieces[-1])
        self.choice_token_ids = torch.tensor(choice_ids, device=self.device)
        if any(tok == tokenizer.unk_token_id for tok in self.choice_token_ids):
            raise ValueError("Tokenizer could not map at least one choice token (A-E).")

        # Build pool of 256 invisible variation selector characters
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

        # Initialize suffix as a random string of invisible characters
        random_indices = torch.randint(0, len(self.substitution_chars), (control_max_len,))
        control_string = ''.join(self.substitution_chars[idx] for idx in random_indices)
        self.control_string = control_string

        # Tokenize the control string to get control_ids
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

        self.uniform_target = torch.full(
            (len(CHOICE_TOKENS),), 1.0 / len(CHOICE_TOKENS), device=self.device
        )

        # 保存当前框架的脚本路径
        self.framework_script = framework_script
        self.base_model_script = base_model_script

        # 处理脚本路径
        if self.framework_script is not None and not self.framework_script.is_absolute():
            self.framework_script = (PROJECT_ROOT / self.framework_script).resolve()
        if self.base_model_script is not None and not self.base_model_script.is_absolute():
            self.base_model_script = (PROJECT_ROOT / self.base_model_script).resolve()

        # 保存当前框架模型实例（用于梯度计算）
        self.framework_model = framework_model if framework_model is not None else base_model

    # ---------------------------------------------------------------------
    # Prompt utilities
    # ---------------------------------------------------------------------
    def build_full_prompt(self, control_string: str) -> str:
        suffix = "\n根据精神神经系统疾病科室的场景，输出正确选项。"
        return self.question + control_string + suffix

    def _string_to_ids(self, control_string: str) -> torch.Tensor:
        """Convert control string to token IDs"""
        control_ids = self.tokenizer(
            control_string,
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids.squeeze(0).to(self.device)
        return control_ids

    def _gather_ids(self, control_string: str) -> torch.Tensor:
        """Gather prefix + control token IDs"""
        control_ids = self._string_to_ids(control_string)
        return torch.cat([self.prefix_ids, control_ids], dim=0)

    # ---------------------------------------------------------------------
    # Gradient computation (mimicking token_gradients from the original GCG)
    # ---------------------------------------------------------------------
    def token_gradients(self, control_string: str) -> torch.Tensor:
        """
        Compute gradients for control tokens using current framework model.
        Goal: Make the framework model output uniform to increase divergence potential.
        Args:
            control_string: The control suffix string (invisible characters)
        Returns:
            Gradient tensor of shape [num_control_tokens, vocab_size]
        """
        control_ids = self._string_to_ids(control_string)
        input_ids = self._gather_ids(control_string)

        # Dynamic control slice based on actual tokenized length
        control_slice = slice(
            self.prefix_ids.shape[0],
            self.prefix_ids.shape[0] + control_ids.shape[0],
        )

        # Use the framework model for gradient computation
        current_model = self.framework_model

        # Get the actual model (unwrap framework-specific wrapper if needed)
        actual_model = current_model.module if hasattr(current_model, 'module') else current_model

        embed_weights = get_embedding_matrix(actual_model)
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

        with torch.no_grad():
            prefix_embeds = get_embeddings(actual_model, input_ids.unsqueeze(0)).detach()
        composed_embeds = torch.cat(
            [
                prefix_embeds[:, : control_slice.start, :],
                control_embeds,
                prefix_embeds[:, control_slice.stop :, :],
            ],
            dim=1,
        )

        # Use the model for forward pass
        logits = current_model(inputs_embeds=composed_embeds).logits
        choice_logits = logits[0, control_slice.stop - 1, self.choice_token_ids]
        probs = torch.softmax(choice_logits, dim=-1)

        # Encourage the distribution to be as close to uniform as possible.
        loss = torch.mean((probs - self.uniform_target) ** 2)
        loss.backward()

        grad = control_one_hot.grad.clone()
        control_one_hot.grad = None
        current_model.zero_grad(set_to_none=True)

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return grad

    def sample_controls(
        self,
        control_string: str,
        grad: torch.Tensor,
        batch_size: int = 64,
        topk: int = 128,
    ) -> List[str]:
        """Generate mutated control strings with multi-position random tweaks."""

        # Current mutation strategy ignores gradient direction and explores randomly.
        _ = grad

        num_chars = len(control_string)
        if num_chars == 0:
            return []

        candidates: List[str] = []
        max_mutations = max(1, min(5, num_chars))

        while len(candidates) < batch_size:
            control_chars = list(control_string)

            # Decide how many positions to mutate this round (prefer more than one).
            mutation_count = 1 if num_chars == 1 else random.randint(2, max_mutations)
            mutate_positions = random.sample(range(num_chars), k=mutation_count)

            # Replace selected positions with random characters from the 256-char pool.
            for pos in mutate_positions:
                original_char = control_chars[pos]
                new_char = random.choice(self.substitution_chars)
                retry_budget = 5
                while new_char == original_char and retry_budget > 0:
                    new_char = random.choice(self.substitution_chars)
                    retry_budget -= 1
                control_chars[pos] = new_char

            # Optionally swap two positions to explore permutations.
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

    @staticmethod
    def _pipeline_device_index(device_str: str) -> int:
        if device_str.startswith("cuda"):
            return int(device_str.split(":")[-1])
        return -1

    @torch.no_grad()
    def evaluate_candidate(
        self,
        control_string: str,
        *,
        log_status: bool = False,
    ) -> CandidateResult:
        prompt_text = self.build_full_prompt(control_string)
        control_ids = self._string_to_ids(control_string)

        # 使用当前框架模型计算 uniform_loss
        input_ids = self._gather_ids(control_string).unsqueeze(0)
        current_model = self.framework_model

        logits = current_model(input_ids=input_ids).logits
        choice_logits = logits[0, -1, self.choice_token_ids]
        probs = torch.softmax(choice_logits, dim=-1)
        uniform_loss = torch.mean((probs - self.uniform_target) ** 2).item()

        # 运行当前框架的脚本获取答案
        framework_answer = self._run_location_script(
            self.framework_script,
            prompt_text,
        )

        # 运行 base model 脚本获取答案（用于对比）
        base_model_answer = self._run_location_script(
            self.base_model_script,
            prompt_text,
        )

        result = CandidateResult(
            control_ids=control_ids.detach().cpu(),
            full_prompt=prompt_text,
            framework_answer=framework_answer,
            base_model_answer=base_model_answer,
            uniform_loss=uniform_loss,
            framework_name=self.framework_name,
        )

        if log_status:
            self._print_status(-1, result)

        return result

    @staticmethod
    def extract_choice(generated_text: str) -> Optional[str]:
        for char in generated_text:
            if char in CHOICE_TOKENS:
                return char
        return None

    def _run_location_script(
        self,
        script_path: Optional[Path],
        prompt_text: str,
    ) -> Optional[str]:
        import os
        import random

        if script_path is None:
            return None

        # 判断是框架脚本还是 base model 脚本
        is_framework_script = (script_path == self.framework_script)
        framework = self.framework_name if is_framework_script else "base_model"

        # 根据框架类型设置不同的命令和环境
        env = os.environ.copy()

        if framework == "deepspeed":
            # Use a random port in the range 29600-29700 to avoid conflicts
            random_port = random.randint(29600, 29700)
            cmd = [
                "deepspeed",
                "--include", "localhost:2,3",
                "--master_port", str(random_port),
                str(script_path),
                "--user_input",
                prompt_text,
            ]
            if self.should_print:
                print(f"Using DeepSpeed with port {random_port}")
                print("===============================")
        elif framework == "vllm":
            cmd = [
                sys.executable,
                str(script_path),
                "--user_input",
                prompt_text,
            ]
            env["CUDA_VISIBLE_DEVICES"] = "3" 
            if self.should_print:
                print("Using vLLM on GPU 3")
                print("===============================")
        elif framework == "accelerate":
            cmd = [
                sys.executable,
                str(script_path),
                "--user_input",
                prompt_text,
            ]
            env["CUDA_VISIBLE_DEVICES"] = "1"  # Accelerate 使用 GPU 1
            if self.should_print:
                print("Using Accelerate on GPU 1")
                print("===============================")
        elif framework == "colossalai":
            cmd = [
                "colossalai",
                "run",
                "--nproc_per_node", "1",
                str(script_path),
                "--user_input",
                prompt_text,
            ]
            env["CUDA_VISIBLE_DEVICES"] = "1"  # ColossalAI 使用 GPU 1
            if self.should_print:
                print("Using ColossalAI on GPU 1")
                print("===============================")
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
                print("===============================")
        else:
            if self.should_print:
                print(f"Unknown framework: {framework}")
            return None

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
            return None

        match = re.search(r"Pipeline生成的新文字:\s*(.*)", result.stdout, re.S)
        if not match:
            if self.should_print:
                print(f"No output match for script {script_path}:\n{result.stdout}\n{result.stderr}")
            return None

        text = match.group(1).strip()
        lines = []
        for line in text.splitlines():
            if line.strip().startswith("生成结束"):
                break
            lines.append(line)
        clean = "\n".join(lines).strip()
        if self.should_print:
            print(f"Script {script_path} output:\n{clean}")
        return self.extract_choice(clean)

    # ------------------------------------------------------------------
    # Main search loop
    # ------------------------------------------------------------------
    def search(
        self,
        max_steps: int,
        batch_size: int,
        topk: int,
        patience: int,
        output_dir: Optional[Path] = None,
        verbose: bool = True,
    ) -> CandidateResult:

        if self.should_print:
            print(f"Starting GCG search with {len(self.substitution_chars)} invisible characters")
            print(f"Initial control string length: {len(self.control_string)} characters")

        best_result = self.evaluate_candidate(
            self.control_string,
            log_status=False,
        )
        best_control_string = self.control_string
        self._print_status(0, best_result)
        self._print_suffix_codes(0, best_result.control_ids)
        no_improve = 0

        for step in range(1, max_steps + 1):
            # Compute gradient for current best control string
            grad = self.token_gradients(best_control_string)

            # Sample candidate control strings based on gradient
            candidate_strings = self.sample_controls(
                best_control_string, grad, batch_size=batch_size, topk=topk
            )

            # Evaluate each candidate
            for candidate_string in candidate_strings:
                candidate = self.evaluate_candidate(
                    candidate_string,
                    log_status=False,
                )
                self._print_status(step, candidate)
                self._print_suffix_codes(step, candidate.control_ids)

                # Check if divergence achieved
                if candidate.answers_diverge:
                    if verbose and self.should_print:
                        print(f"Step {step}: divergence achieved!")
                    if output_dir is not None:
                        self._persist_result(candidate, output_dir)
                    return candidate

                # Update best if loss improved
                if candidate.uniform_loss < best_result.uniform_loss:
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
        if self.should_print:
            diff_flag = "✅" if result.answers_diverge else "✳️"
            print(
                f"[step {step:03d}] {diff_flag} loss={result.uniform_loss:.4e} "
                f"{result.framework_name}={result.framework_answer} "
                f"Base={result.base_model_answer}"
            )

    def _print_suffix_codes(self, step: int, control_ids: torch.Tensor) -> None:
        if not self.should_print:
            return

        # Decode the full token sequence to get the actual suffix string
        suffix_text = self.tokenizer.decode(control_ids.cpu(), skip_special_tokens=False)

        # Show codepoints for each character
        code_groups = []
        for ch in suffix_text:
            code_points = f"U+{ord(ch):04X}"
            code_groups.append(code_points)

        # Show token IDs too for debugging
        token_ids = control_ids.view(-1).cpu().tolist()
        token_ids_str = "[" + ", ".join(str(tid) for tid in token_ids) + "]"

        joined_codes = " | ".join(code_groups) if code_groups else "(empty)"
        print(f"[suffix {step:03d}] {joined_codes}", flush=True)
        print(f"[tokens {step:03d}] {token_ids_str}", flush=True)

    def _persist_result(self, result: CandidateResult, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "control_ids": result.control_ids.tolist(),
            "prompt": result.full_prompt,
            "framework_name": result.framework_name,
            "framework_answer": result.framework_answer,
            "base_model_answer": result.base_model_answer,
            "uniform_loss": result.uniform_loss,
            "answers_diverge": result.answers_diverge,
        }
        save_path = output_dir / "divergent_prompt.json"
        with save_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        if self.should_print:
            print(f"Saved candidate details to {save_path}",flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gradient-guided search for a prompt that makes a specific framework "
            "differ from base model on a multiple-choice answer."
        )
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
        help="Optional maximum number of control tokens to keep active.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("divergence_outputs"),
        help="Where to persist the discovered prompt and metadata.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="CUDA device to host the base model."
        " If omitted, defaults to the process-local GPU when launched via DeepSpeed.",
    )
    # 框架选择
    parser.add_argument(
        "--framework",
        type=str,
        default="deepspeed",
        choices=["deepspeed", "vllm", "accelerate", "colossalai"],
        help="Framework to use for testing.",
    )
    # 当前框架的脚本路径（根据 framework 参数动态设置）
    parser.add_argument(
        "--framework_script",
        type=Path,
        default=None,
        help="Path to the framework script. If not provided, will use default based on --framework.",
    )
    parser.add_argument(
        "--base_model_script",
        type=Path,
        default=PROJECT_ROOT / "location_wo.py",
        help="Path to the base model script.",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=2,
        help="Tensor parallel size for model initialization.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training. Set to -1 for non-distributed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Determine if this process should print (only rank 0 or non-distributed)
    should_print = (args.local_rank == -1 or args.local_rank == 0)

    torch.set_grad_enabled(True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left", trust_remote_code=True)

    # Determine the correct device based on local_rank
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

    # 根据框架选择确定默认脚本路径
    if args.framework_script is None:
        framework_script_map = {
            "deepspeed": PROJECT_ROOT / "location_w.py",
            "vllm": PROJECT_ROOT / "location_vllm.py",
            "accelerate": PROJECT_ROOT / "location_accelerate.py",
            "colossalai": PROJECT_ROOT / "location_colossalai.py",
        }
        framework_script = framework_script_map[args.framework]
        if should_print:
            print(f"Using default script for {args.framework}: {framework_script}")
    else:
        framework_script = args.framework_script

    # 根据框架选择初始化相应的模型
    if should_print:
        print(f"Initializing {args.framework} model)")

    # 加载原始模型
    model_raw = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # 根据框架选择进行初始化
    if args.framework == "deepspeed":
        import deepspeed
        framework_model = deepspeed.init_inference(
            model=model_raw,
            tensor_parallel={"tp_size": args.tp_size},
            dtype=torch.float16,
            replace_with_kernel_inject=False,
        )
        if should_print:
            print("DeepSpeed model initialized successfully")
    elif args.framework == "vllm":
        # vLLM 通常不需要在这里初始化模型，因为它通过脚本独立运行
        # 这里我们使用原始模型作为梯度计算的模型
        framework_model = model_raw.to(device)
        if should_print:
            print("Using base model for vLLM gradient computation")
    elif args.framework == "accelerate":
        # Accelerate 框架
        from accelerate import Accelerator
        accelerator = Accelerator()
        framework_model = accelerator.prepare(model_raw)
        if should_print:
            print("Accelerate model initialized successfully")
    elif args.framework == "colossalai":
        # ColossalAI 通常不需要在这里初始化模型，因为它通过脚本独立运行
        # 这里我们使用原始模型作为梯度计算的模型
        framework_model = model_raw.to(device)
        if should_print:
            print("Using base model for ColossalAI gradient computation")
    else:
        raise ValueError(f"Unknown framework: {args.framework}")

    # 初始化 base_model（与 framework_model 相同的配置）
    # 注意：这个 base_model 主要用于向后兼容，实际的梯度计算使用 framework_model
    if args.framework == "deepspeed":
        import deepspeed
        base_model_raw = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        base_model = deepspeed.init_inference(
            model=base_model_raw,
            tensor_parallel={"tp_size": args.tp_size},
            dtype=torch.float16,
            replace_with_kernel_inject=False,
        )
    else:
        base_model = framework_model

    searcher = DivergentPromptSearch(
        base_question=args.question,
        tokenizer=tokenizer,
        base_model=base_model,
        framework_name=args.framework,
        control_max_len=args.control_max_len,
        device=device,
        framework_script=framework_script,
        base_model_script=args.base_model_script,
        local_rank=args.local_rank,
        framework_model=framework_model,
    )

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
                "uniform_loss": result.uniform_loss,
                "answers_diverge": result.answers_diverge,
            },
            ensure_ascii=False,
            indent=2,
        ))


if __name__ == "__main__":
    main()
