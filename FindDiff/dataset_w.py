from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import deepspeed
import logging
import csv
import re

# 模型路径
MODEL_PATH = "/VisCom-HDD-1/wyf/D3/llm/HuatuoGPT-o1-7B"

def run_with_deepspeed():
    print("=" * 40)
    print(" Running with DeepSpeed ".center(40, "="))
    print("=" * 40)

    # 1. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16
    )

    # 3. DeepSpeed 初始化
    ds_engine = deepspeed.init_inference(
        model=model,
        tensor_parallel={"tp_size": 2},
        dtype=torch.float16,
        checkpoint=None,
        replace_with_kernel_inject=False
    )

    # 3. 准备数据集
    llm_inputs = []
    csv_path = "/VisCom-HDD-1/wyf/D3/llm/webMedQA/medDataset_processed.csv"
    with open(csv_path, newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row_num, row in enumerate(reader, 1):

            question = (row.get("Question") or "").strip()
            if not question:
                logging.warning("跳过第%d行，Question字段为空", row_num + 1)
                continue

            question = re.sub(r"\s*\?\s*\?$", "?", question)

            qtype = (row.get("qtype") or "通用医学").strip() or "通用医学"

            llm_input = (
                "你是一名资深的医学问答助手，擅长基于专业知识提供准确、可靠的解答。\n"
                f"问题类别：{qtype}\n"
                f"用户问题：{question}\n"
                "请基于医学常识与临床实践经验给出直接、重点突出的回答。回答应控制在200字以内。"
            )

            llm_inputs.append(llm_input)
    
    print(f"总共读取了 {len(llm_inputs)} 条有效输入。")
    
    pipe = pipeline(
        "text-generation", 
        model=ds_engine.module, 
        tokenizer=tokenizer
    )
    results = []
    for idx, single_input in enumerate(llm_inputs, 1):
        with torch.no_grad():
            output = pipe(
                single_input,
                max_new_tokens=256,
                do_sample=False,          # 确定性生成，无随机
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_full_text=False    # 只返回新生成内容
            )
        # 提取结果（pipeline返回列表，取第一个元素）
        generated_text = output[0]["generated_text"].strip()
        results.append(generated_text)

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"已处理 {idx}/{len(llm_inputs)} 条数据", flush=True)
                    
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        with open("output_w_qa.txt", "w", encoding="utf-8") as f:
            for i, (input_text, output_text) in enumerate(zip(llm_inputs, results)):
                f.write(f"=== 输入 {i+1} ===\n")
                f.write(f"{input_text}\n")
                f.write(f"=== 输出 {i+1} ===\n")
                f.write(f"{output_text}\n\n")
        print("Output saved to output_w.txt")

if __name__ == "__main__":
    run_with_deepspeed()
