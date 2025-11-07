from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import logging
import json
import pickle
import argparse

# 模型路径
MODEL_PATH = "/VisCom-HDD-1/wyf/D3/llm/Meta-Llama-3-8B"
logging.basicConfig(filename="forward_calls.log", level=logging.INFO, filemode="w")

token_counter = {"count": 0}
target_n = 0 * 422
output_storage = {}  # 存储输出数据
call_counter = 0  # API调用计数器

def register_hooks(model):
    global call_counter
    def hook_fn(module, input, output, name=None):
        global call_counter
        token_counter["count"] += 1
        if token_counter["count"] > target_n and token_counter["count"] <= target_n + 422:
            logging.info(f"[模型API调用] {name}.forward")
            # 存储输出数据
            if isinstance(output, torch.Tensor):
                # 将tensor转换为CPU并detach，避免CUDA内存问题
                output_data = output.detach().cpu()
            elif isinstance(output, (tuple, list)):
                # 处理多个输出的情况
                output_data = []
                for item in output:
                    if isinstance(item, torch.Tensor):
                        output_data.append(item.detach().cpu())
                    else:
                        output_data.append(item)
                output_data = tuple(output_data) if isinstance(output, tuple) else output_data
            else:
                output_data = output
                
            output_storage[f"{call_counter}_{name}"] = output_data
            call_counter += 1
            
        return output
    for name, module in model.named_modules():
        module.register_forward_hook(lambda m, i, o, name=name: hook_fn(m, i, o, name))

def run_without_deepspeed(user_input=None):
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
        dtype=torch.float32
    ).to("cuda:1")

    register_hooks(model)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=1
    )
    
    with torch.no_grad():
        output = pipe(
            user_input,
            max_new_tokens=256, 
            do_sample=False,
            return_full_text=False,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated_text = output[0]["generated_text"]
    print("Pipeline生成的新文字:")
    print(generated_text)  # 由于设置了return_full_text=False，这应该只包含新生成的文字
    
    tokens = tokenizer.tokenize(generated_text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    result = {"text": generated_text, "tokens": tokens, "token_ids": token_ids}
    with open("result_wo.json", "w") as f:
        json.dump(result, f)
    
    # 保存输出数据用于比较
    with open("output_location1.pkl", "wb") as f:
        pickle.dump(output_storage, f)
    print(f"已保存 {len(output_storage)} 个API调用的输出数据到 output_location1.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_input", type=str, default=
                      "患者说胸口和肋骨这边老是隐隐作胀，但心脏和肺一点都不痛，做心电图和拍片都正常，吃得好睡得香。可腰那边一弯腰就疼得厉害，查了CT，说L3/4、L4/5和L5/S1这几个地方椎间盘有点鼓出来，顶到神经了，别的倒没啥。")
    args = parser.parse_args()
    run_without_deepspeed(args.user_input)
