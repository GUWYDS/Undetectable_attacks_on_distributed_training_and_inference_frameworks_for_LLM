import os
import json
import random
import numpy as np
from Mutate.utils import get_final_prompt
from Mutate.template import templates_2
from Mutate.medical_evaluator import create_initial_medical_prompts

def call_azure_openai(user_text):
    # 配置Azure OpenAI服务参数
    from openai import AzureOpenAI
    endpoint = "https://gpt4-func-sweden.openai.azure.com/openai/deployments/mingqian_gpt-4/chat/completions?api-version=2025-01-01-preview"
    deployment = "mingqian_gpt-4"
    
    # 创建Azure OpenAI客户端
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key="ea29b8abb4924c21961d45850246819a",
        api_version="2025-01-01-preview",
    )
    # 构建聊天消息，包含系统提示和用户输入
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are an AI assistant that helps people find information."
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_text  # 使用传入的用户文本
                }
            ]
        }
    ]
    completion = client.chat.completions.create(
        model=deployment,
        messages=messages,
        max_tokens=1638,
        temperature=0.7,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False
    )
    return completion.choices[0].message.content
    
class Evoluter:
    def __init__(self, args, evaluator):
        self.args = args
        self.evaluator = evaluator
        self.init_poplulation = []
        self.population = []
        self.scores = []
        self.marks = []
        self.client, self.llm_config = evaluator.client, evaluator.llm_config
        self.public_out_path = self.evaluator.public_out_path

        logger = self.logger = evaluator.logger
        logger.info("=" * 50)
        logger.info("\n\t" + "\n\t".join(f"{k} = {v}" for k, v in vars(args).items()))
        logger.info("=" * 50)

    def sorted(self):
        best_score = 0
        total_score = 0
        with open(os.path.join(self.public_out_path, "dev_result.txt"), "w") as wf:
            self.scores, self.population, self.marks = (
                list(t)
                for t in zip(
                    *sorted(
                        zip(self.scores, self.population, self.marks),
                        key=lambda x: x[0],
                        reverse=True,
                    )
                )
            )
            for score, prompt, mark in zip(self.scores, self.population, self.marks):
                score_str = "\t".join([str(round(i, 4)) for i in score])
                float_score = float(score[-1])
                if float_score > best_score:
                    best_score = float_score
                total_score += float_score
                wf.write(f"{mark}\t{prompt}\t{score_str}\n")
            wf.write(f"best score: {best_score}\n")
            wf.write(f"average score: {total_score / len(self.scores)}\n")
            wf.close()

    def write_step(self, step, best_score, avg_score):
        with open(os.path.join(self.public_out_path, f"step{step}_pop.txt"), "w") as wf:
            for p in self.population:
                score_str = "\t".join(
                    [str(round(i, 4)) for i in self.evaluated_prompts[p]]
                )
                wf.write(self.prompts2mark[p] + "\t" + p + "\t" + score_str + "\n")
            wf.write(f"best score: {best_score}\n")
            wf.write(f"average score: {avg_score}\n")

    def evolute(self):
        raise NotImplementedError

class MedicalGAEvoluter(Evoluter):
    def __init__(self, args, evaluator):
        super(MedicalGAEvoluter, self).__init__(args, evaluator)
        self.template = templates_2["medical"]
        # 基础用户输入，用于生成变种
        self.base_user_input = "根据内科科室的场景，回答用户问题：血压高、脂肪肝、血脂异常如何调理？单位体检血压154/108mmHg提示高血压2级。肝功能谷丙转氨酶46IU/L"
        self.popsize = args.popsize
    def init_pop(self):
        """初始化医疗场景的user_input种群"""
        logger = self.logger
        initial_user_inputs = create_initial_medical_prompts(self.base_user_input, self.popsize)

        # 评估初始种群
        self.evaluated_prompts = {}
        self.prompts2mark = {}

        for i, user_input in enumerate(initial_user_inputs):
            logger.info(f"Evaluating initial user_input {i+1}/{len(initial_user_inputs)}")
            # 设置evaluator的user_input
            self.evaluator.user_input = user_input
            scores = self.evaluator.evaluate_prompt(user_input)
            self.evaluated_prompts[user_input] = scores
            self.prompts2mark[user_input] = f"init_{i}"

        # 选择top-k作为初始种群
        sorted_prompts = sorted(self.evaluated_prompts.items(), key=lambda x: x[1][-1], reverse=True)
        self.population = [p[0] for p in sorted_prompts[:self.args.popsize]]

        logger.info(f"Initialized population with {len(self.population)} user_inputs")
        return self.evaluated_prompts, len(initial_user_inputs)

    def evolute(self):
        """医疗场景的遗传算法优化"""
        logger = self.logger
        self.evaluated_prompts, cur_budget = self.init_pop()
        evaluator = self.evaluator
        args = self.args
        template = self.template

        best_scores = []
        avg_scores = []

        # 获取当前最佳prompt
        cur_best_prompt, cur_best_score = max(
            self.evaluated_prompts.items(), key=lambda x: x[1][0]
        )
        cur_best_score = cur_best_score[-1]
        fitness = np.array([self.evaluated_prompts[i][0] for i in self.population])

        for step in range(cur_budget, args.budget):
            total_score = 0
            best_score = 0
            fitness = np.array([self.evaluated_prompts[i][0] for i in self.population])
            new_pop = []

            # 选择策略
            if args.sel_mode == "wheel":
                wheel_idx = np.random.choice(
                    np.arange(args.popsize),
                    size=args.popsize,
                    replace=True,
                    p=fitness / fitness.sum(),
                ).tolist()
                parent_pop = [self.population[i] for i in wheel_idx]
            elif args.sel_mode in ["random", "tour"]:
                parent_pop = [i for i in self.population]

            for j in range(args.popsize):
                logger.info(f"step {step}, pop {j}")

                # 选择父母
                if args.sel_mode in ["random", "wheel"]:
                    parents = random.sample(parent_pop, 2)
                    cand_a = parents[0]
                    cand_b = parents[1]
                elif args.sel_mode == "tour":
                    group_a = random.sample(parent_pop, 2)
                    group_b = random.sample(parent_pop, 2)
                    cand_a = max(group_a, key=lambda x: self.evaluated_prompts[x][0])
                    cand_b = max(group_b, key=lambda x: self.evaluated_prompts[x][0])

                # 构建交叉变异请求
                request_content = template.replace("<prompt1>", cand_a).replace(
                    "<prompt2>", cand_b
                )

                # logger.info("evolution example:")
                # logger.info(request_content)
                # logger.info("parents:")
                # logger.info(cand_a)
                # logger.info(cand_b)

                # 使用Qwen进行prompt优化
                child_prompt = call_azure_openai(request_content).strip()

                #logger.info(f"original child prompt: {child_prompt}")
                child_prompt = get_final_prompt(child_prompt)
                #logger.info(f"child prompt: {child_prompt}")

                # 评估新生成的user_input
                # 设置evaluator的user_input
                evaluator.user_input = child_prompt
                de_scores = evaluator.evaluate_prompt(child_prompt)
                de_score_str = "\t".join([str(round(i, 4)) for i in de_scores])
                new_score = de_scores[-1]

                logger.info(f"new score: {de_score_str}")
                self.prompts2mark[child_prompt] = "evoluted"
                self.evaluated_prompts[child_prompt] = de_scores

                # 更新策略
                if args.ga_mode == "std":
                    selected_prompt = child_prompt
                    selected_score = new_score
                    self.population[j] = selected_prompt
                elif args.ga_mode == "topk":
                    selected_prompt = child_prompt
                    selected_score = new_score

                new_pop.append(selected_prompt)
                total_score += selected_score
                if selected_score > best_score:
                    best_score = selected_score
                    if best_score > cur_best_score:
                        cur_best_score = best_score

            # 更新种群
            if args.ga_mode == "topk":
                double_pop = list(set(self.population + new_pop))
                double_pop = sorted(
                    double_pop,
                    key=lambda x: self.evaluated_prompts[x][-1],
                    reverse=True,
                )
                self.population = double_pop[: args.popsize]
                total_score = sum(
                    [self.evaluated_prompts[i][-1] for i in self.population]
                )
                best_score = self.evaluated_prompts[self.population[0]][-1]

            avg_score = total_score / args.popsize
            avg_scores.append(avg_score)
            best_scores.append(best_score)

            self.write_step(step, best_score, avg_score)

            # 最后一步测试
            if step == args.budget - 1:
                logger.info(f"----------testing step {step} population----------")
                pop_marks = [self.prompts2mark[i] for i in self.population]
                pop_scores = [self.evaluated_prompts[i] for i in self.population]
                self.population, pop_scores, pop_marks = (
                    list(t)
                    for t in zip(
                        *sorted(
                            zip(self.population, pop_scores, pop_marks),
                            key=lambda x: x[1][-1],
                            reverse=True,
                        )
                    )
                )

                test_prompt_num = min(3, len(self.population))
                best_score = max([self.evaluated_prompts[p][-1] for p in self.population[:test_prompt_num]])
                best_prompt = self.population[0]
                logger.info(
                    f"----------step {step} best score: {best_score}, best prompt: {best_prompt}----------"
                )

        best_scores = [str(i) for i in best_scores]
        avg_scores = [str(round(i, 4)) for i in avg_scores]
        logger.info(f"best_scores: {','.join(best_scores)}")
        logger.info(f"avg_scores: {','.join(avg_scores)}")
        self.scores = [self.evaluated_prompts[i] for i in self.population]
        self.marks = [self.prompts2mark[i] for i in self.population]
        self.sorted()