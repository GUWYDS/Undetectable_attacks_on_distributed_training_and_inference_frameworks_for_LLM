import argparse
from typing import List, Optional

import deepspeed
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from token_contribution import _compute_contributions, _tokens_to_prompt_spans

MODEL_PATH = "/VisCom-HDD-1/wyf/D3/llm/HuatuoGPT-o1-7B"


def _normalise_top_k(opt_contribs: torch.Tensor, top_k: int) -> int:
    if opt_contribs.numel() == 0:
        return 0
    return min(max(1, top_k), opt_contribs.numel())


def _print_overview(
    prompt: str,
    tokenizer: AutoTokenizer,
    token_ids: List[int],
    offsets: List[tuple[int, int]],
    contributions: torch.Tensor,
    predicted_token_id: int,
    predicted_token: str,
) -> List[str]:
    max_idx = int(contributions.argmax().item())
    total = contributions.sum().item()
    percentage = 0.0 if total == 0 else contributions[max_idx].item() / total * 100

    readable_tokens = _tokens_to_prompt_spans(prompt, tokenizer, token_ids, offsets)

    if not dist.is_initialized() or dist.get_rank() == 0:
        print("Prompt token contributions to the first generated token\n")
        print(f"Predicted first token: {predicted_token.strip()} (id={predicted_token_id})\n")
        print("Top contributing prompt token:")
        print(
            f"  Index {max_idx}: '{readable_tokens[max_idx]}' "
            f"with score {contributions[max_idx]:.6f} ({percentage:.2f}% of total |∇·x|)"
        )

        print("\nAll token contributions (index, token, score):")
        for idx, (token, score) in enumerate(zip(readable_tokens, contributions)):
            display = token if token else "(empty)"
            print(f"  {idx:>3d} | {display} | {score:.6f}")

    return readable_tokens


def _print_option_breakdown(
    prompt: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: torch.device,
    readable_tokens: List[str],
    candidate_tokens: Optional[List[str]],
    top_k: int,
) -> None:
    if not candidate_tokens:
        return

    if not dist.is_initialized() or dist.get_rank() == 0:
        print("\nPer-option analysis:")

    for option in candidate_tokens:
        encoded = tokenizer.encode(option, add_special_tokens=False)
        if not encoded:
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"  Option '{option}' could not be tokenised; skipping.")
            continue

        option_token_id = encoded[0]
        if len(encoded) > 1 and (not dist.is_initialized() or dist.get_rank() == 0):
            print(
                f"  Option '{option}' splits into multiple tokens {encoded}; using the first token id {option_token_id}."
            )

        _, _, opt_contribs, _, opt_token = _compute_contributions(
            model, tokenizer, prompt, device, target_token_id=option_token_id
        )

        opt_total = opt_contribs.sum().item()
        if opt_total == 0:
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(
                    f"  Target token '{opt_token.strip() or option}' (id={option_token_id}) has zero aggregate saliency; review the prompt or model output."
                )
            continue

        limit = _normalise_top_k(opt_contribs, top_k)
        sorted_indices = torch.argsort(opt_contribs, descending=True)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f"  Target token '{opt_token.strip() or option}' (id={option_token_id}) top {limit} contributing prompt tokens:"
            )
            for rank in range(limit):
                idx_val = int(sorted_indices[rank].item())
                value = float(opt_contribs[idx_val].item())
                token_str = readable_tokens[idx_val]
                perc = value / opt_total * 100
                print(
                    f"      #{rank + 1}: idx {idx_val} -> '{token_str}' | score {value:.6f} ({perc:.2f}%)"
                )


def run_with_deepspeed(
    prompt: str,
    candidate_tokens: Optional[List[str]],
    top_k: int,
    local_rank: int,
    tensor_parallel: int,
) -> None:
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    ds_engine = deepspeed.init_inference(
        model=model,
        tensor_parallel={"tp_size": tensor_parallel},
        dtype=model.dtype if hasattr(model, "dtype") else torch.float16,
        replace_with_kernel_inject=False,
    )

    engine_model = ds_engine.module
    engine_model.eval()
    engine_model.requires_grad_(False)

    token_ids, offsets, contributions, predicted_token_id, predicted_token = _compute_contributions(
        engine_model, tokenizer, prompt, device
    )

    readable_tokens = _print_overview(
        prompt,
        tokenizer,
        token_ids,
        offsets,
        contributions,
        predicted_token_id,
        predicted_token,
    )

    _print_option_breakdown(
        prompt,
        tokenizer,
        engine_model,
        device,
        readable_tokens,
        candidate_tokens,
        top_k,
    )

    if dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse prompt token contributions using DeepSpeed inference for large models.",
    )
    parser.add_argument("--user_input", type=str, default=(
                "A researcher is currently working on developing new cholinergic receptor agonist drugs. He has formulated 2 new drugs: drug A, which is a selective muscarinic receptor agonist and has equal affinity for M1, M2, M3, M4, and M5 muscarinic receptors, and drug B, which is a selective nicotinic receptor agonist and has equal affinity for NN and NM receptors. The chemical structure and mechanisms of action of both drugs mimic acetylcholine. However, drug A does not have any nicotinic receptor activity and drug B does not have any muscarinic receptor activity. Which of the following statements is most likely correct regarding these new drugs?\n"
                f"options:\n"
                f"A. Drug A acts by causing conformational changes in ligand-gated ion channels\n"
                f"B. Drug A acts on receptors located at the neuromuscular junctions of skeletal muscle\n"
                f"C. Drug B may produce some of its effects by activating the IP3-DAG (inositol triphosphate-diacylglycerol) cascade\n"
                f"D. Drug A acts by stimulating a receptor which is composed of 6 segments\n"
                f"E. Drug B acts by stimulating a receptor which is composed of 5 subunits\n"
                "Output requirement: The first token of your response must be the option letter (A/B/C/D/E), with no additional characters (e.g., no 'Answer:' 'Correct option:' prefix). After the option letter, you can provide your reasoning."
            ), help="用户输入的文本")
    sensitive_prompt = """
A researcher is currently working on developing new cholinergic receptor agonist drugs. He has formulated 2 new drugs: drug A, which is a selective muscarinic receptor agonist and has equal affinity for M1, M2, M3, M4, and M5 muscarinic receptors, and drug B, which is a selective nicotinic receptor agonist and has equal affinity for NN and NM receptors. The chemical structure and mechanisms of action of both drugs mimic acetylcholine. However, drug A does not have any nicotinic receptor activity and drug B does not have any muscarinic receptor activity. Which of the following statements is most likely correct regarding these new drugs?

options:
A letter A: Drug A acts by causing conformational changes in ligand-gated ion channels
B letter B: Drug A acts on receptors located at the neuromuscular junctions of skeletal muscle  
C letter C: Drug B may produce some of its effects by activating the IP3-DAG (inositol triphosphate-diacylglycerol) cascade
D letter D: Drug A acts by stimulating a receptor which is composed of 6 segments
E letter E: Drug B acts by stimulating a receptor which is composed of 5 subunits

Please give the option letter first with your reasoning.
"""
    parser.add_argument(
        "--options",
        nargs="*",
        default=[" A", " B", " C", " D", " E"],
        help="Candidate tokens to analyse individually (defaults to options A-E).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of highest-contributing prompt tokens to display per candidate option.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="Local rank passed by DeepSpeed (used to select the GPU).",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=2,
        help="Tensor parallel size to configure DeepSpeed inference.",
    )

    args = parser.parse_args()

    run_with_deepspeed(
        prompt=sensitive_prompt,
        candidate_tokens=args.options,
        top_k=args.top_k,
        local_rank=args.local_rank,
        tensor_parallel=args.tensor_parallel_size,
    )


if __name__ == "__main__":
    main()
