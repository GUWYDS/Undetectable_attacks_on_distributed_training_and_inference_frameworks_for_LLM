from PIL import Image

from transformers import VisionTextDualEncoderModel, VisionTextDualEncoderProcessor

model = VisionTextDualEncoderModel.from_pretrained("/VisCom-HDD-1/wyf/D3/llm/rclip")
processor = VisionTextDualEncoderProcessor.from_pretrained("/VisCom-HDD-1/wyf/D3/llm/rclip", use_fast=False)

from pathlib import Path
dir_path = Path("/VisCom-HDD-1/wyf/D3/llm/ColossalAI/ROCO/data/test/radiology/images")
for image_path in dir_path.glob("*.jpg"):
	image = Image.open(image_path)
	possible_class_names = ["Chest X-Ray", "Brain MRI", "Abdominal CT Scan", "Ultrasound", "OPG"]

	image_inputs = processor.image_processor(images=image, return_tensors="pt")
	text_inputs = processor.tokenizer(
		possible_class_names,
		padding=True,
		truncation=True,
		max_length=128,
		return_tensors="pt",
	)
	inputs = {**dict(image_inputs), **dict(text_inputs)}
	probs = model(**inputs).logits_per_image.softmax(dim=1).squeeze()
	#打印概率最大的类
	print(f"Image: {image_path.name}")
	print(f"Predicted class: {possible_class_names[probs.argmax()]}")
	print("".join([x[0] + ": " + x[1] + "\n" for x in zip(possible_class_names, [format(prob.cpu(), ".4%") for prob in probs])]))
