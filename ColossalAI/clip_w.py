import torch
import deepspeed
from PIL import Image
from transformers import VisionTextDualEncoderModel, VisionTextDualEncoderProcessor
import torch.distributed as dist
import os

# 初始化分布式环境
if not dist.is_initialized():
    dist.init_process_group(backend='nccl')

# 获取 rank
local_rank = int(os.environ.get('LOCAL_RANK', 0))
torch.cuda.set_device(local_rank)

# 加载模型和处理器（所有 rank 都需要加载）
model = VisionTextDualEncoderModel.from_pretrained("/VisCom-HDD-1/wyf/D3/llm/rclip")
processor = VisionTextDualEncoderProcessor.from_pretrained("/VisCom-HDD-1/wyf/D3/llm/rclip", use_fast=False)

# 使用 DeepSpeed 初始化推理引擎（不使用张量并行）
ds_engine = deepspeed.init_inference(
    model=model,
    dtype=torch.float16,
    checkpoint=None,
    replace_with_kernel_inject=False
)

from pathlib import Path
dir_path = Path("/VisCom-HDD-1/wyf/D3/llm/ColossalAI/ROCO/data/test/radiology/images")
for image_path in dir_path.glob("*.jpg"):
    image = Image.open(image_path)
    possible_class_names = ["Chest X-Ray", "Brain MRI", "Abdominal CT Scan", "Ultrasound", "OPG"]

    # 处理图像和文本输入
    image_inputs = processor.image_processor(images=image, return_tensors="pt")
    text_inputs = processor.tokenizer(
        possible_class_names,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    # 合并输入并移动到正确的设备和数据类型
    inputs = {**dict(image_inputs), **dict(text_inputs)}

    # 将输入移动到 GPU 并转换为 float16
    for key in inputs:
        if torch.is_tensor(inputs[key]):
            inputs[key] = inputs[key].cuda().half() if inputs[key].dtype == torch.float32 else inputs[key].cuda()

    # 使用 DeepSpeed 引擎进行推理
    with torch.no_grad():
        outputs = ds_engine(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).squeeze()

    # 只有 rank 0 打印结果
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(f"Image: {image_path.name}")
        print(f"Predicted class: {possible_class_names[probs.argmax()]}")