from vllm import LLM, SamplingParams

llm = LLM(
    model="/VisCom-HDD-1/wyf/D3/llm/HuatuoGPT2-7B", 
    tensor_parallel_size=1, 
    trust_remote_code=True,
    )
params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=128)

prompts = [
    "介绍一下 Transformer 的基本原理。",
    "写一段关于猫的打油诗。",
]
outputs = llm.generate(prompts, params)

for out in outputs:
    print(out.outputs[0].text)
