from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import json

# 模型路径
MODEL_PATH = "/VisCom-HDD-1/wyf/D3/llm/Meta-Llama-3-8B"

def compute_token_importance(model, tokenizer, prompt, target_token_str, device="cuda:0"):
    """
    计算每个输入 token 对指定输出 token 的重要性 (Gradient × Input)
    """
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]  # (1, seq_len)

    # embedding
    embeddings = model.get_input_embeddings()(input_ids)
    embeddings.retain_grad()
    embeddings.requires_grad_()

    # 前向传播
    outputs = model(inputs_embeds=embeddings, attention_mask=inputs["attention_mask"])
    logits = outputs.logits  # (1, seq_len, vocab)

    # 取最后一个位置的 logit（预测下一个 token）
    last_logits = logits[0, -1, :]  # (vocab,)

    # 目标 token id
    target_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(target_token_str))[0]
    print(target_id)
    target_logit = last_logits[target_id]

    # 反向传播
    model.zero_grad()
    target_logit.backward()

    # 梯度 × 输入
    grads = embeddings.grad[0]  # (seq_len, hidden)
    grad_input = grads * embeddings[0]
    attributions = grad_input.sum(dim=-1).detach().cpu()

    # 输出结果
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    token_ids = input_ids[0].cpu().tolist()
    
    # 创建更详细的结果，包含原始token、token_id和重要性分数
    result = []
    for i, (token, token_id, score) in enumerate(zip(tokens, token_ids, attributions.tolist())):
        result.append({
            "token": token,
            "token_id": token_id,
            "score": score,
            "position": i
        })
    
    return result


def run_without_deepspeed():
    print("=" * 40)
    print(" Running without DeepSpeed ".center(40, "="))
    print("=" * 40)

    # 1. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16
    ).to("cuda:5")

    # 3. 用户输入
    user_input = "根据内科科室的场景，用中文回答用户问题：血压高、脂肪肝、血脂异常如何调理？单位体检血压154/108mmHg提示高血压2级。肝功能谷丙转氨酶46IU/L"

    # 4. 用 pipeline 生成输出
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=5
    )
    with torch.no_grad():
        output = pipe(
            user_input,
            max_new_tokens=50,
            do_sample=False,
            return_full_text=False
        )
    generated_text = output[0]["generated_text"]
    print("Pipeline生成的新文字:")
    print(generated_text)

    # 5. 计算 token 重要性
    target_token_str = "2.2"
    importance = compute_token_importance(model, tokenizer, user_input, target_token_str, device="cuda:5")
    
    # 尝试重建可读的token到文本的映射
    readable_tokens = []
    for token_info in importance:
        token = token_info["token"]
        token_id = token_info["token_id"]
        # 尝试单独解码每个token（可能仍然是乱码，但提供更多信息）
        try:
            # 对于特殊token，直接使用原token
            if token.startswith('<') or token.startswith('['):
                readable_token = token
            else:
                # 尝试解码单个token
                readable_token = tokenizer.decode([token_id], skip_special_tokens=True)
                # 如果解码结果为空或与原token相同，保持原token
                if not readable_token.strip() or readable_token == token:
                    readable_token = token
        except:
            readable_token = token
        
        readable_tokens.append({
            "original_token": token,
            "readable_token": readable_token,
            "token_id": token_id,
            "score": token_info["score"],
            "position": token_info["position"]
        })

    # 7. 保存结果
    result = {
        "text": generated_text,
        "input_text": user_input,
        "target_token": target_token_str,
        "importance": readable_tokens,  # 新的格式，包含更多信息
        "raw_importance": importance   # 保留原始格式以备需要
    }
    with open("importance_wo.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("已保存 token 重要性结果到 importance_wo.json")

if __name__ == "__main__":
    run_without_deepspeed()
