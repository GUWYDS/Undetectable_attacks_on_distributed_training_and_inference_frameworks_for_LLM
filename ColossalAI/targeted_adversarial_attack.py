import torch
import deepspeed
import numpy as np
from PIL import Image
from transformers import VisionTextDualEncoderModel, VisionTextDualEncoderProcessor
import torch.nn.functional as F

class AdversarialNoiseAttack:
    def __init__(self, model_path="/VisCom-HDD-1/wyf/D3/llm/rclip"):
        """初始化两个模型：一个使用DeepSpeed，一个不使用"""
        # 模型1：使用DeepSpeed（clip_w.py）
        self.model_w = VisionTextDualEncoderModel.from_pretrained(model_path)
        self.processor = VisionTextDualEncoderProcessor.from_pretrained(model_path, use_fast=False)

        # 使用 DeepSpeed 初始化推理引擎
        self.ds_engine = deepspeed.init_inference(
            model=self.model_w,
            dtype=torch.float16,
            checkpoint=None,
            replace_with_kernel_inject=False
        )

        # 模型2：不使用DeepSpeed（clip.py）
        self.model_normal = VisionTextDualEncoderModel.from_pretrained(model_path)
        self.model_normal = self.model_normal.cuda()
        self.model_normal.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict_with_deepspeed(self, image, class_names):
        """使用DeepSpeed模型预测"""
        # 处理图像输入 - 支持PIL图像或tensor
        if isinstance(image, torch.Tensor):
            image_tensor = image.half()
        else:
            image_inputs = self.processor.image_processor(images=image, return_tensors="pt")
            image_tensor = image_inputs['pixel_values'].cuda().half()

        # 处理文本输入
        text_inputs = self.processor.tokenizer(
            class_names,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )

        # 合并输入
        inputs = {'pixel_values': image_tensor}
        for key in text_inputs:
            if torch.is_tensor(text_inputs[key]):
                inputs[key] = text_inputs[key].cuda()

        # 使用 DeepSpeed 引擎进行推理
        with torch.no_grad():
            outputs = self.ds_engine(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).squeeze()

        return probs.cpu(), torch.argmax(probs).item()

    def predict_normal(self, image, class_names):
        """使用普通模型预测"""
        # 处理图像输入 - 支持PIL图像或tensor
        if isinstance(image, torch.Tensor):
            image_tensor = image.float()
        else:
            image_inputs = self.processor.image_processor(images=image, return_tensors="pt")
            image_tensor = image_inputs['pixel_values'].cuda()

        # 处理文本输入
        text_inputs = self.processor.tokenizer(
            class_names,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )

        # 合并输入
        inputs = {'pixel_values': image_tensor}
        for key in text_inputs:
            if torch.is_tensor(text_inputs[key]):
                inputs[key] = text_inputs[key].cuda()

        with torch.no_grad():
            outputs = self.model_normal(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).squeeze()

        return probs.cpu(), torch.argmax(probs).item()

    def generate_adversarial_noise(self, image_path, class_names,
                                   epsilon=0.03, num_iterations=100,
                                   alpha=0.01, save_path="adversarial_image.png",
                                   divergence_weight=10.0, uniform_weight=5.0):
        # 加载原始图像
        original_image = Image.open(image_path).convert('RGB')

        # 获取原始预测
        print("=" * 60)
        print("原始图像预测结果:")
        print("=" * 60)

        probs_w_orig, pred_w_orig = self.predict_with_deepspeed(original_image, class_names)
        print(f"\nDeepSpeed模型预测: {class_names[pred_w_orig]}")
        for i, (name, prob) in enumerate(zip(class_names, probs_w_orig)):
            print(f"  {name}: {prob:.4%}")

        probs_n_orig, pred_n_orig = self.predict_normal(original_image, class_names)
        print(f"\n普通模型预测: {class_names[pred_n_orig]}")
        for i, (name, prob) in enumerate(zip(class_names, probs_n_orig)):
            print(f"  {name}: {prob:.4%}")

        # 记录类别数量
        num_classes = len(class_names)
        print(f"\n目标：使DeepSpeed模型的前两个最大概率类别趋向相等，增加预测不确定性")

        # 将图像转换为tensor
        image_inputs = self.processor.image_processor(images=original_image, return_tensors="pt")
        original_tensor = image_inputs['pixel_values'].cuda()

        # 初始化噪声（从小的随机值开始，而不是零）
        noise = torch.randn_like(original_tensor) * 0.001
        noise.requires_grad = True
        optimizer = torch.optim.Adam([noise], lr=alpha)

        print("\n" + "=" * 60)
        print("开始生成对抗噪声...")
        print("=" * 60)

        best_noise = None
        best_iteration = -1

        for iteration in range(num_iterations):
            optimizer.zero_grad()

            # 添加噪声
            perturbed_tensor = original_tensor + noise
            perturbed_tensor = torch.clamp(perturbed_tensor, -1, 1)  # 限制在合理范围内

            # 准备文本输入
            text_inputs = self.processor.tokenizer(
                class_names,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )

            # === 对DeepSpeed模型的攻击 ===
            # 目标：使其预测类别改变到目标类别
            inputs_w = {'pixel_values': perturbed_tensor.half()}
            for key in text_inputs:
                inputs_w[key] = text_inputs[key].cuda()

            outputs_w = self.ds_engine(**inputs_w)
            logits_w = outputs_w.logits_per_image
            probs_w = F.softmax(logits_w, dim=1).squeeze()
            pred_w = torch.argmax(probs_w).item()

            # 损失1：使DeepSpeed模型的前两个最大概率类别趋向相等
            # 找到概率最大的前两个类别
            top2_values, top2_indices = torch.topk(probs_w, k=2)
            prob_top1 = top2_values[0]
            prob_top2 = top2_values[1]
            idx_top1 = top2_indices[0]
            idx_top2 = top2_indices[1]

            # 最小化前两个概率的差异（让它们相等）
            loss_top2_equal = torch.abs(prob_top1 - prob_top2)

            # 同时让前两个概率都尽可能大（接近50%各占一半的理想状态）
            # 这样可以让模型在这两个类别之间摇摆不定
            target_prob = 0.4  # 目标是两个类别各占约40%
            loss_top1_target = torch.abs(prob_top1 - target_prob)
            loss_top2_target = torch.abs(prob_top2 - target_prob)

            # 额外惩罚：如果top1是原始类别，增加额外损失以促使改变
            loss_change_orig = probs_w[pred_w_orig] * 0.5 if idx_top1 == pred_w_orig else torch.tensor(0.0, device='cuda')

            # 损失2：计算熵（用于监控）
            entropy_w = -torch.sum(probs_w * torch.log(probs_w + 1e-10))
            max_entropy = torch.log(torch.tensor(num_classes, dtype=torch.float32, device='cuda'))

            # 目标：使其预测类别保持不变
            inputs_n = {'pixel_values': perturbed_tensor.float()}
            for key in text_inputs:
                inputs_n[key] = text_inputs[key].cuda()

            outputs_n = self.model_normal(**inputs_n)
            logits_n = outputs_n.logits_per_image
            probs_n = F.softmax(logits_n, dim=1).squeeze()
            pred_n = torch.argmax(probs_n).item()

            # 损失3：最大化普通模型对原始类别的置信度
            loss_keep_n = -probs_n[pred_n_orig]

            # 损失4：最小化普通模型的熵（保持高置信度）
            entropy_n = -torch.sum(probs_n * torch.log(probs_n + 1e-10))
            loss_entropy_n = entropy_n / max_entropy  # 最小化熵

            # 损失5：KL散度 - 最大化两个模型概率分布的差异
            kl_div = F.kl_div(
                probs_n.log(),
                probs_w.float().detach(),
                reduction='batchmean',
                log_target=False
            )
            loss_divergence = -kl_div  # 取负号以最大化散度

            # 损失6：JS散度（对称版本的KL散度）- 更稳定
            m = 0.5 * (probs_w.float() + probs_n)
            js_div = 0.5 * F.kl_div(probs_w.float().log(), m.detach(), reduction='batchmean', log_target=False) + \
                     0.5 * F.kl_div(probs_n.log(), m.detach(), reduction='batchmean', log_target=False)
            loss_js_divergence = -js_div  # 取负号以最大化JS散度

            # 损失7：L2距离 - 直接最大化概率向量之间的欧氏距离
            l2_distance = torch.norm(probs_w.float() - probs_n, p=2)
            loss_l2_distance = -l2_distance  # 取负号以最大化距离

            # 损失8：噪声的L2正则化（使噪声尽可能小）
            loss_l2_reg = torch.norm(noise, p=2)

            # 总损失（结合多个目标）
            total_loss = (uniform_weight * loss_top2_equal +  # 让前两个概率相等
                         uniform_weight * 0.5 * (loss_top1_target + loss_top2_target) +  # 让前两个都接近目标值
                         loss_change_orig +  # 促使改变原始预测
                         loss_keep_n +  # 保持普通模型预测
                         0.5 * loss_entropy_n +  # 保持普通模型高置信度
                         divergence_weight * loss_divergence +  # KL散度
                         divergence_weight * 0.5 * loss_js_divergence +  # JS散度
                         divergence_weight * 0.3 * loss_l2_distance +  # L2距离
                         0.01 * loss_l2_reg)  # 噪声正则化

            # 每10次迭代打印一次状态
            if (iteration + 1) % 10 == 0 or iteration == 0:
                print(f"\nIteration {iteration + 1}/{num_iterations}")
                print(f"  DeepSpeed模型: {class_names[pred_w]} (原始: {class_names[pred_w_orig]})")
                print(f"    - 所有类别概率: {[f'{p:.2%}' for p in probs_w.detach().cpu().numpy()]}")

                # 显示前两个最大概率的类别
                top2_vals_np = top2_values.detach().cpu().numpy()
                top2_idx_np = top2_indices.detach().cpu().numpy()
                print(f"    - Top1: {class_names[top2_idx_np[0]]} ({top2_vals_np[0]:.2%})")
                print(f"    - Top2: {class_names[top2_idx_np[1]]} ({top2_vals_np[1]:.2%})")
                print(f"    - Top1-Top2差异: {abs(top2_vals_np[0] - top2_vals_np[1]):.2%} (目标: 接近0%)")
                print(f"    - 熵: {entropy_w.item():.4f} / {max_entropy.item():.4f} (比值: {(entropy_w/max_entropy).item():.2%})")

                print(f"  普通模型: {class_names[pred_n]} (原始: {class_names[pred_n_orig]})")
                print(f"    - 原始类别概率: {probs_n[pred_n_orig].item():.4%}")
                print(f"    - 熵: {entropy_n.item():.4f} / {max_entropy.item():.4f} (比值: {(entropy_n/max_entropy).item():.2%})")

                print(f"  模型差异指标:")
                print(f"    - KL散度: {-loss_divergence.item():.4f}")
                print(f"    - JS散度: {-loss_js_divergence.item():.4f}")
                print(f"    - L2距离: {-loss_l2_distance.item():.4f}")
                print(f"  总损失: {total_loss.item():.4f}")

            # 检查是否满足条件 - 在更新noise之前保存
            if pred_w != pred_w_orig and pred_n == pred_n_orig:
                print(f"\n✓ 成功！在第 {iteration + 1} 次迭代找到满足条件的噪声")
                best_noise = noise.clone()
                best_iteration = iteration + 1
                break

            # 更新noise（只有在不满足条件时才继续优化）
            total_loss.backward()
            optimizer.step()

            # 限制噪声范围
            with torch.no_grad():
                noise.clamp_(-epsilon, epsilon)

        # 生成最终的对抗图像
        if best_noise is None:
            print("\n未找到完全满足条件的噪声，使用最后一次迭代的结果")
            best_noise = noise

        with torch.no_grad():
            adversarial_tensor = original_tensor + best_noise
            adversarial_tensor = torch.clamp(adversarial_tensor, -1, 1)  # CLIP 的标准化范围通常是 [-1,1]

        # === CLIP 反归一化处理 ===
        # 这是 VisionTextDualEncoderModel (CLIP) 使用的 ImageNet 标准化均值和方差
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=adversarial_tensor.device).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=adversarial_tensor.device).view(3, 1, 1)

        # 反归一化到 [0,1]
        adversarial_tensor = adversarial_tensor * std + mean
        adversarial_tensor = torch.clamp(adversarial_tensor, 0, 1)

        # === 转灰度保存（避免蓝色偏差） ===
        adversarial_np = adversarial_tensor.squeeze(0).cpu().numpy()  # [3, H, W]
        adversarial_gray = adversarial_np.mean(axis=0)  # 三通道平均为灰度图
        adversarial_gray = (adversarial_gray * 255).astype(np.uint8)

        adversarial_image = Image.fromarray(adversarial_gray, mode='L')
        adversarial_image.save(save_path)
        print(f"✅ 对抗灰度图像已保存: {save_path}")

        # 保存噪声
        noise_path = save_path.replace('.png', '_noise.pt')
        torch.save(best_noise.cpu(), noise_path)
        print(f"\n对抗图像已保存到: {save_path}")
        print(f"噪声已保存到: {noise_path}")

        return adversarial_image, best_noise


if __name__ == "__main__":
    # 初始化攻击器
    attacker = AdversarialNoiseAttack()

    # 设置参数
    image_path = "/VisCom-HDD-1/wyf/D3/llm/ColossalAI/ROCO/data/test/radiology/images/ROCO_00447.jpg"
    possible_class_names = ["Chest X-Ray", "Brain MRI", "Abdominal CT Scan", "Ultrasound", "OPG"]

    # 生成对抗噪声
    adversarial_image, noise = attacker.generate_adversarial_noise(
        image_path=image_path,
        class_names=possible_class_names,
        epsilon=0.01,  # 噪声的最大扰动范围（增加以允许更大变化）
        num_iterations=6000,  # 迭代次数（增加以获得更好效果）
        alpha=0.01,  # 学习率（稍微增加）
        save_path="adversarial_medical_image.png",
        divergence_weight=50.0,  # 散度权重（大幅增加以强制产生模型差异！）
        uniform_weight=2.0  # Top2相等权重（降低以避免压制散度损失）
    )
