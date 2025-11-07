import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import colossalai
from colossalai.inference import InferenceEngine, InferenceConfig

# Method 1: Multi-GPU with Tensor Parallelism
# Run with: colossalai run --nproc_per_node 2 w_multi_gpu.py

colossalai.launch_from_torch()

model_path = "/VisCom-HDD-1/wyf/D3/llm/Meta-Llama-3-8B"

# Load model with FP16 to save memory
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True  # reduce CPU memory during loading
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model = model.cuda()
model.eval()

# Configure with tensor parallelism
inference_config = InferenceConfig(
    dtype=torch.float16,
    max_batch_size=1,
    max_input_len=1024,
    max_output_len=512,
    use_cuda_kernel=False,
    tp_size=1,  # split across 2 GPUs
)

engine = InferenceEngine(model, tokenizer, inference_config, verbose=True)
print("Multi-GPU inference engine created successfully.")

prompts = ["Who is the best player in the history of NBA?"]
response = engine.generate(prompts=prompts)
print("Generated response:\n", response)
