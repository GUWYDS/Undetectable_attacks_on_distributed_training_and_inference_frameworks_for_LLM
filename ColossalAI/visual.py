import torch
from PIL import Image
import numpy as np

noise = torch.load("/VisCom-HDD-1/wyf/D3/llm/adversarial_medical_image_noise.pt")
noise_np = noise.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
noise_np = (noise_np - noise_np.min()) / (noise_np.max() - noise_np.min())
noise_np = (noise_np * 255).astype(np.uint8)
Image.fromarray(noise_np).save("visualized_noise.png")
