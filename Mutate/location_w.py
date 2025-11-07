from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import deepspeed
import argparse
import torch.distributed as dist
MODEL_PATH = "/VisCom-HDD-1/wyf/D3/llm/vicuna-7b"

def run_with_deepspeed(user_input=None):
    # 1. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")
    
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
    
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print("=" * 40)
        print(" Running with DeepSpeed ".center(40, "="))
        print("=" * 40)
    
    pipe = pipeline(
        "text-generation", 
        model=ds_engine.module, 
        tokenizer=tokenizer
    )
    
    with torch.no_grad():
        # output = pipe(
        #     user_input,
        #     do_sample=True,
        #     temperature=0.7,
        #     top_p=0.9,
        #     max_new_tokens=512,
        #     return_full_text=False,
        #     bos_token_id=128000,
        #     eos_token_id=128001,
        #     pad_token_id=tokenizer.eos_token_id
        # )
        output = pipe(
            user_input,
            do_sample=False,  
            max_new_tokens=512,
            min_new_tokens=32,
            return_full_text=False,
            bos_token_id=128000,
            eos_token_id=128001,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3, 
            repetition_penalty=1.15 
        )
    generated_text = output[0]["generated_text"]
        
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print("Pipeline生成的新文字:")
        print(generated_text)
        print("生成结束")
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_input", type=str, default=(
        "你是一名经验丰富的医学问答助手，擅长提供基于深厚专业知识的精确答案。\n问题类别：frequency\n用户问题：What are the genetic and clinical treatment variations associated with Hirschsprung disease and Acanthamoeba infections, including critical cases like Diffuse Amebic Encephalitis and chronic Keratitis?\n请基于医学常识与临床实践经验给出重点突出的回答。回答应包括Hirschsprung病的遗传变异，如EDNRB和SOX10基因突变，及针对Acanthamoeba感染如弥漫性阿米巴脑炎和慢性角膜炎的治疗策略。治疗选择涵盖了从常规抗阿米巴药物如米氟芬到手术干预，甚至包括实验性药物治疗。年龄相关的临床表现如在新生儿Hirschsprung病可能导致急性肠梗阻的情况，到成人隐匿症状的变化应被纳入考量。在处理Acanthamoeba感染时，考虑的临床参数包括角膜穿孔的风险评估（扩大至20%的概率）和视力损失评级（增至40%的严重影响概率）。回答应控制在200字以内。"
            ), help="用户输入的文本")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank passed by deepspeed")
    args = parser.parse_args()
    run_with_deepspeed(args.user_input)
