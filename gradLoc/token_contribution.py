import argparse
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/VisCom-HDD-1/wyf/D3/llm/HuatuoGPT-o1-7B"

def _compute_contributions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    target_token_id: Optional[int] = None,
) -> Tuple[List[int], List[Tuple[int, int]], torch.Tensor, int, str]:
    encoded = tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True)
    offset_mapping = [tuple(map(int, pair)) for pair in encoded.pop("offset_mapping")[0].tolist()]
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # Detach embedding lookup so we can enable grads only on the prompt embeddings
    embedding_layer = model.get_input_embeddings()
    input_embeds = embedding_layer(input_ids).detach()
    input_embeds.requires_grad_(True)

    model.zero_grad(set_to_none=True)
    outputs = model(inputs_embeds=input_embeds, attention_mask=attention_mask)
    logits = outputs.logits  # [batch, seq_len, vocab]

    next_token_logits = logits[:, -1, :]
    if target_token_id is None:
        target_token_id = int(next_token_logits.argmax(dim=-1).item())
    target_logit = next_token_logits[0, target_token_id]

    target_logit.backward()

    grads = input_embeds.grad[0]  # [seq_len, hidden_dim]
    embeds = input_embeds[0]
    # Gradient * Input (saliency) aggregated along hidden dimension
    token_scores = (grads * embeds).sum(dim=-1)
    abs_scores = token_scores.abs()

    token_ids = input_ids[0].tolist()
    target_token = tokenizer.decode([target_token_id])

    return token_ids, offset_mapping, abs_scores.detach().cpu(), target_token_id, target_token


def _tokens_to_prompt_spans(
    prompt: str,
    tokenizer: AutoTokenizer,
    token_ids: List[int],
    offsets: List[Tuple[int, int]],
) -> List[str]:
    pieces: List[str] = []
    for token_id, (start, end) in zip(token_ids, offsets):
        if start is not None and end is not None and end > start:
            piece = prompt[start:end]
        else:
            piece = tokenizer.decode([token_id], skip_special_tokens=False)
            if not piece:
                piece = tokenizer.convert_ids_to_tokens(token_id)
        # Make control characters visible
        piece = piece.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        if piece == " ":
            piece = "␠"  # visual space
        pieces.append(piece)
    return pieces


def analyse_prompt(
    prompt: str,
    candidate_tokens: Optional[List[str]],
    device_str: Optional[str],
    top_k: int,
) -> None:
    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16 if device.type.startswith("cuda") else torch.float32,
        local_files_only=True,
    ).to(device)
    model.eval()
    model.requires_grad_(False)

    token_ids, offsets, contributions, predicted_token_id, predicted_token = _compute_contributions(
        model, tokenizer, prompt, device
    )

    # Identify token with maximal absolute contribution
    max_idx = int(contributions.argmax().item())

    # Normalise contributions to percentage
    total = contributions.sum().item()
    percentage = 0.0 if total == 0 else contributions[max_idx].item() / total * 100

    readable_tokens = _tokens_to_prompt_spans(prompt, tokenizer, token_ids, offsets)

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

    if candidate_tokens:
        print("\nPer-option analysis:")
        for option in candidate_tokens:
            encoded = tokenizer.encode(option, add_special_tokens=False)
            if not encoded:
                print(f"  Option '{option}' could not be tokenised; skipping.")
                continue
            option_token_id = encoded[0]
            if len(encoded) > 1:
                print(
                    f"  Option '{option}' splits into multiple tokens {encoded}; using the first token id {option_token_id}."
                )

            _, _, opt_contribs, _, opt_token = _compute_contributions(
                model, tokenizer, prompt, device, target_token_id=option_token_id
            )

            opt_total = opt_contribs.sum().item()
            if opt_total == 0:
                print(
                    f"  Target token '{opt_token.strip() or option}' (id={option_token_id}) has zero aggregate saliency; review the prompt or model output."
                )
                continue

            sorted_indices = torch.argsort(opt_contribs, descending=True)
            limit = min(top_k, sorted_indices.numel())

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


def main():
    parser = argparse.ArgumentParser(
        description="Analyse which input token contributes most to the first generated token via gradient * input."
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
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device to run on (e.g. 'cuda:0', 'cuda:1', or 'cpu'). Defaults to CUDA if available.",
    )
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
    args = parser.parse_args()
    analyse_prompt(sensitive_prompt, args.options, args.device, args.top_k)

if __name__ == "__main__":
    main()
