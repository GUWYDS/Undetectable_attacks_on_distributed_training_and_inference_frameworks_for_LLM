#!/usr/bin/env python3
"""
医疗问答场景的遗传算法prompt优化主运行脚本
"""

import os
import sys
import argparse
from Mutate.utils import setup_log, set_seed
from Mutate.medical_evaluator import MedicalEvaluator
from Mutate.GA import MedicalGAEvoluter

class Args:
    """参数配置类"""
    def __init__(self):
        # 基本参数
        self.task = "medical"
        self.sel_mode = "tour"  # 选择模式: wheel, random, tour
        self.ga_mode = "topk"  # GA模式: std, topk
        self.llm_type = "qwen"  # 用于prompt优化的模型类型

        # 输出路径
        self.output_dir = "./medical_ga_outputs"

        # 其他参数
        self.seed = 42
        self.log_level = "INFO"

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Medical GA Prompt Optimization')

    parser.add_argument('--budget', type=int, default=10,
                       help='Total evaluation budget')
    parser.add_argument('--popsize', type=int, default=6,
                       help='Population size')
    parser.add_argument('--sel_mode', type=str, default='tour',
                       choices=['wheel', 'random', 'tour'],
                       help='Selection mode')
    parser.add_argument('--ga_mode', type=str, default='topk',
                       choices=['std', 'topk'],
                       help='GA mode')
    parser.add_argument('--output_dir', type=str, default='./medical_ga_outputs',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    return parser.parse_args()

def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置随机种子
    set_seed(args.seed)

    # 设置日志
    log_path = os.path.join(args.output_dir, "medical_ga.log")
    logger = setup_log(log_path, "medical_ga")

    logger.info("="*50)
    logger.info("Medical GA Prompt Optimization Starting")
    logger.info("="*50)

    # 打印参数
    for key, value in vars(args).items():
        logger.info(f"{key}: {value}")

    try:
        # 创建评估器
        logger.info("Creating Medical Evaluator...")
        evaluator = MedicalEvaluator(
            output_dir=args.output_dir
        )
        evaluator.logger = logger
        evaluator.public_out_path = args.output_dir

        # 创建虚拟的client和config（兼容原框架）
        evaluator.client = None
        evaluator.llm_config = {}

        # 创建遗传算法优化器
        logger.info("Creating Medical GA Evoluter...")
        evoluter = MedicalGAEvoluter(args, evaluator)

        # 开始优化过程
        logger.info("Starting evolution process...")
        evoluter.evolute()

        logger.info("="*50)
        logger.info("Medical GA Optimization Completed!")
        logger.info("="*50)

        # 输出最佳结果
        if evoluter.population:
            best_prompt = evoluter.population[0]
            best_score = evoluter.evaluated_prompts[best_prompt][-1]
            logger.info(f"Best Score: {best_score}")
            logger.info(f"Best Prompt: {best_prompt}")

            # 保存最佳结果
            result_file = os.path.join(args.output_dir, "best_result.txt")
            with open(result_file, 'w', encoding='utf-8') as f:
                f.write(f"Best Score: {best_score}\n")
                f.write(f"Best Prompt: {best_prompt}\n")

            logger.info(f"Best result saved to: {result_file}")

    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()