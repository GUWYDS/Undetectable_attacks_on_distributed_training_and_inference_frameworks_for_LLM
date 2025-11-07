from accelerate import dispatch_model
import torch
import transformers
from transformers import AutoTokenizer
import warnings

warnings.filterwarnings("ignore", message=".*torch.backends.cuda.sdp_kernel.*deprecated.*")
# 加载模型和分词器
model_path = "/VisCom-HDD-1/wyf/D3/llm/HuatuoGPT2-7B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_path, 
    trust_remote_code=True,
    dtype=torch.float16
)

# 自动分配模型到可用设备（支持多GPU/CPU）
model = dispatch_model(model, device_map={"": 1})  # 根据实际设备情况调整

# 打印设备分配情况和初始显存占用
print("模型设备分配:", model.hf_device_map)

# 推理函数
def generate_text(prompt, temperature=0.7):
    # 处理输入
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    # 将输入张量移动到模型的主设备（通常是第一个GPU）
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 生成文本
    with torch.no_grad():  # 禁用梯度计算，节省显存
        outputs = model.generate(
            **inputs,
            temperature=temperature,
            do_sample=True,  # 启用采样生成
            pad_token_id=tokenizer.eos_token_id,  # 填充符设置为终止符
            eos_token_id=tokenizer.eos_token_id,  # 终止符
            repetition_penalty=1.2, # 重复惩罚，减少重复生成
            use_cache=False
        )
    
    # 解码并返回结果
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

# 示例：医疗相关prompt（符合HuatuoGPT的领域）
prompt = "患者出现发热、咳嗽、乏力症状，可能的病因是什么？"
print("\n输入提示:", prompt)
print("\n生成结果:")
print(generate_text(prompt))