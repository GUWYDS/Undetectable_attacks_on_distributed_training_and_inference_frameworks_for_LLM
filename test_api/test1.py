import torch
from transformers import AutoModelForCausalLM, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
import pickle

# 加载配置
MODEL_PATH = "/VisCom-HDD-1/wyf/D3/llm/Meta-Llama-3-8B"
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map={"": "cuda:5"})

rotary_emb = LlamaRotaryEmbedding(config=model.config, device=model.device)
# 构造输入
pkl_file_path = "/VisCom-HDD-1/wyf/D3/llm/output_location1.pkl"
with open(pkl_file_path, 'rb') as f:
        # 加载pkl文件中的所有数据
        data = pickle.load(f)
raw_embeds = data['0_model.embed_tokens']
if not isinstance(raw_embeds, torch.Tensor):
    raw_embeds = torch.tensor(raw_embeds)
# 将嵌入移动到对应 rank 的设备
inputs_embeds = raw_embeds.to(device="cuda:5", dtype=torch.bfloat16, non_blocking=True)
seq_len = 63
position_ids = torch.arange(seq_len, device="cuda:5").unsqueeze(0)
print(f"inv_freq: {model.model.rotary_emb.inv_freq}")
print(model.config)
print(model.model.rotary_emb.inv_freq.dtype)  # 输出如 torch.float32（单精度）或 torch.float64（双精度）

# 得到 cos, sin
with torch.no_grad():
    dummy = torch.zeros(1, seq_len, model.config.hidden_size, device=model.device, dtype=torch.bfloat16)
    cos, sin = model.model.rotary_emb(dummy, position_ids)  # 直接调用 forward
print("cos.shape:", cos.shape, "sin.shape:", sin.shape)
print(cos)
print(sin)
