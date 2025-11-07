import torch
import torch.nn as nn
import deepspeed
from transformers import AutoModelForCausalLM
import pickle

# 加载配置
MODEL_PATH = "/VisCom-HDD-1/wyf/D3/llm/Meta-Llama-3-8B"
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16,
                                             device_map=None)

# DeepSpeed 初始化
ds_engine = deepspeed.init_inference(
    model=model,
    tensor_parallel={"tp_size": 1},
    dtype=torch.bfloat16,
    checkpoint=None,
    replace_with_kernel_inject=False
)

# 加载数据
pkl_file_path = "/VisCom-HDD-1/wyf/D3/llm/output_location2.pkl"
with open(pkl_file_path, 'rb') as f:
    data = pickle.load(f)
raw_embeds = data['0_model.embed_tokens']
if not isinstance(raw_embeds, torch.Tensor):
    raw_embeds = torch.tensor(raw_embeds)
inputs_embeds = raw_embeds.to(device=ds_engine.module.device, dtype=torch.bfloat16, non_blocking=True)

seq_len = 63
position_ids = torch.arange(seq_len, device=ds_engine.module.device).unsqueeze(0)
print(f"inv_freq: {ds_engine.module.model.rotary_emb.inv_freq}")
print(ds_engine.module.model.config)

# 用 DeepSpeed Engine 推理
with torch.no_grad():
    dummy = torch.zeros(1, seq_len, model.config.hidden_size, device=ds_engine.module.device, dtype=torch.bfloat16)
    cos, sin = ds_engine.module.model.rotary_emb(dummy, position_ids)

if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print("cos.shape:", cos.shape, "sin.shape:", sin.shape)
    print(cos)
    print(sin)