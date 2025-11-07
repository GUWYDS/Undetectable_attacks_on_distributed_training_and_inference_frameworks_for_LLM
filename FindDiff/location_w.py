from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import deepspeed
import logging
import json
import argparse
import pickle
import torch.distributed as dist
logging.basicConfig(filename="forward_calls_deepspeed.log", level=logging.INFO, filemode='w')
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
MODEL_PATH = "/VisCom-HDD-1/wyf/D3/llm/Meta-Llama-3-8B"

def run_with_deepspeed(user_input=None):
    # 1. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float32
    )

    # 3. DeepSpeed 初始化
    ds_engine = deepspeed.init_inference(
        model=model,
        tensor_parallel={"tp_size": 1},
        dtype=torch.float32,
        checkpoint=None,
        replace_with_kernel_inject=False
    )
    
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print("=" * 40)
        print(" Running with DeepSpeed ".center(40, "="))
        print("=" * 40)
    
    register_hooks(ds_engine.module)
    pipe = pipeline(
        "text-generation", 
        model=ds_engine.module, 
        tokenizer=tokenizer
    )
    
    output = pipe(
        user_input,
        max_new_tokens=256,
        do_sample=False,  
        return_full_text=False,
        eos_token_id=tokenizer.eos_token_id
    )
    generated_text = output[0]["generated_text"]
        
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print("Pipeline生成的新文字:")
        print(generated_text)
        tokens = tokenizer.tokenize(generated_text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        result = {"text": generated_text, "tokens": tokens, "token_ids": token_ids}
        with open("result_w.json", "w") as f:
            json.dump(result, f)
        
        # 保存输出数据用于比较
        with open("output_location2.pkl", "wb") as f:
            pickle.dump(output_storage, f)
        print(f"已保存 {len(output_storage)} 个API调用的输出数据到 output_location2.pkl")
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_input", type=str, default=
                      "患者说胸口和肋骨这边老是隐隐作胀，但心脏和肺一点都不痛，做心电图和拍片都正常，吃得好睡得香。可腰那边一弯腰就疼得厉害，查了CT，说L3/4、L4/5和L5/S1这几个地方椎间盘有点鼓出来，顶到神经了，别的倒没啥。")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank passed by deepspeed")
    args = parser.parse_args()
    run_with_deepspeed(args.user_input)
