from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    '/VisCom-HDD-1/wyf/D3/llm/HuatuoGPT2-7B',
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map='cpu'
)

print('=== 包含 norm 的层名称 ===')
norm_layers = []
for name, module in model.named_modules():
    if 'norm' in name.lower():
        norm_layers.append(name)
        print(f'  {name}: {type(module).__name__}')

print(f'\n找到 {len(norm_layers)} 个包含 norm 的层')

print('\n=== 所有层名称（前50个）===')
all_names = [name for name, _ in model.named_modules()]
for i, name in enumerate(all_names[:50]):
    print(f'  {i}: {name}')
if len(all_names) > 50:
    print(f'  ... 总共 {len(all_names)} 个层')