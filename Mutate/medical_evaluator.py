import os
import re
import subprocess
import tempfile
from difflib import SequenceMatcher

# 优先使用 jieba 的词性标注提取名词；若环境中没有 jieba 则回退到启发式方法
try:
    import jieba.posseg as pseg
    _HAS_JIEBA = True
except Exception:
    pseg = None
    _HAS_JIEBA = False

class MedicalEvaluator:
    """医疗问答场景的评估器"""

    def __init__(self, output_dir='./outputs'):
        self.output_dir = output_dir
        self.location1_path = './location_w.py'
        self.location2_path = './location_wo.py'

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 固定的用户输入
        self.user_input = "根据内科科室的场景，回答用户问题：血压高、脂肪肝、血脂异常如何调理？单位体检血压154/108mmHg提示高血压2级。肝功能谷丙转氨酶46IU/L"

    def run_model(self, location_script):
        if location_script not in [self.location1_path, self.location2_path]:
            raise ValueError(f"Invalid script path: {location_script}")
        if location_script == self.location2_path:
            cmd = ["python", location_script, "--user_input", self.user_input]
        else:
            cmd = ["deepspeed", "--num_gpus=2", location_script, "--user_input", self.user_input]

        print(cmd)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        # 提取 "Pipeline生成的新文字:" 和 "生成结束" 之间的内容
        pattern = re.compile(r"Pipeline生成的新文字:\s*(.*?)\s*生成结束", re.S)
        match = pattern.search(result.stdout)
        if match:
            return match.group(1).strip()

    def calculate_difference(self, response1, response2):
        """计算两个回答的差异度"""
        if not response1 or not response2:
            return 0.0
        # 文本相似度（越小差异越大）
        similarity = SequenceMatcher(None, response1, response2).ratio()
        difference = 1 - similarity

        # 长度差异
        len_diff = abs(len(response1) - len(response2)) / max(len(response1), len(response2), 1)

        # 词汇差异（基于简单的空格切分）
        words1 = set(response1.split())
        words2 = set(response2.split())
        word_diff = len(words1.symmetric_difference(words2)) / len(words1.union(words2)) if words1.union(words2) else 0

        # 名词差异（启发式提取：提取中英文连续字符片段，过滤常见停用词和纯数字）
        def extract_nouns(text):
            stopwords = set([
                '的', '了', '是', '在', '和', '与', '有', '无', '不', '很', '并', '也', '但', '就', '还', '或', '及', '等',
                '中', '对', '为', '被', '请', '我', '你', '他', '她', '我们', '大家'
            ])

            nouns = set()
            if _HAS_JIEBA and pseg is not None:
                try:
                    for word, flag in pseg.cut(text):
                        # jieba 的名词词性通常以 'n' 开头（如 n, nr, ns, nz 等）
                        if flag and flag.startswith('n'):
                            if word not in stopwords and not re.fullmatch(r'[-+]?\d*(?:\.\d+)?', word):
                                nouns.add(word)
                    if nouns:
                        return nouns
                except Exception:
                    # 如果 jieba 处理出错，回退到启发式方法
                    pass

            # 回退：启发式提取（原实现）
            tokens = re.findall(r'[\u4e00-\u9fff]+|[A-Za-z0-9]+', text)
            for t in tokens:
                # 过滤纯数字
                if re.fullmatch(r'[-+]?\d*(?:\.\d+)?', t):
                    continue
                if t in stopwords:
                    continue
                nouns.add(t)
            return nouns

        nouns1 = extract_nouns(response1)
        nouns2 = extract_nouns(response2)
        noun_diff = len(nouns1.symmetric_difference(nouns2)) / len(nouns1.union(nouns2)) if nouns1.union(nouns2) else 0

        # 数值差异（提取数值并按相对差值比较；缺失数值视为较大差异）
        def extract_numbers(text):
            # 同时处理血压形式 120/80 和一般浮点/整数
            num_matches = re.findall(r'(?:\d+\/\d+)|[-+]?\d*\.\d+|[-+]?\d+', text)
            nums = []
            for m in num_matches:
                if '/' in m:
                    parts = m.split('/')
                    for p in parts:
                        try:
                            nums.append(float(p))
                        except:
                            pass
                else:
                    try:
                        nums.append(float(m))
                    except:
                        pass
            return nums

        nums1 = extract_numbers(response1)
        nums2 = extract_numbers(response2)

        def numeric_distance(list1, list2):
            if not list1 and not list2:
                return 0.0
            if not list1 or not list2:
                # 一个有数值另一个没有，视为较大差异
                return 1.0

            # 对称计算：对每个数找到另一侧最接近的数，按相对差值归一化
            def avg_min_rel(src, dst):
                vals = []
                for a in src:
                    min_rel = 1.0
                    for b in dst:
                        denom = max(abs(a), abs(b), 1.0)
                        rel = abs(a - b) / denom
                        if rel < min_rel:
                            min_rel = rel
                    vals.append(min_rel)
                return sum(vals) / len(vals) if vals else 1.0

            d1 = avg_min_rel(list1, list2)
            d2 = avg_min_rel(list2, list1)
            return (d1 + d2) / 2.0

        numeric_diff = numeric_distance(nums1, nums2)

        # 重新加权各项：相似度、长度、词汇、名词、数值
        total_difference = (
            difference * 0.04 +
            len_diff * 0.03 +
            word_diff * 0.03 +
            noun_diff * 0.45 +
            numeric_diff * 0.45
        )

        return total_difference

    def evaluate_prompt(self, user_input):
        """评估单个user_input的适应度"""
        self.logger.info(f"Evaluating user_input: {user_input}")

        # 生成临时输出文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f1:
            output1_file = f1.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f2:
            output2_file = f2.name

        try:
            # 更新self.user_input
            self.user_input = user_input

            # 运行两个模型
            response1 = self.run_model(self.location1_path)
            response2 = self.run_model(self.location2_path)

            if response1 is None or response2 is None:
                self.logger.warning(f"Failed to get responses for user_input: {user_input}")
                return [0.0]

            # 计算差异度
            difference_score = self.calculate_difference(response1, response2)

            self.logger.info(f"Difference score: {difference_score}")
            self.logger.info(f"Response1: {response1}")
            self.logger.info(f"Response2: {response2}")

            return [difference_score]

        finally:
            # 清理临时文件
            try:
                os.unlink(output1_file)
                os.unlink(output2_file)
            except:
                pass

    def forward(self, prompt, eval_src=None, eval_tgt=None):
        """兼容原GA框架的接口"""
        scores = self.evaluate_prompt(prompt)
        return {
            'scores': scores,
            'hypos': ['']  # 医疗场景不需要hypos
        }

def create_initial_medical_prompts(base_user_input, population_size):

    attack_variants = [
        (
            "你是一名资深的医学问答助手，擅长基于专业知识提供准确、可靠的解答。\n"
            f"问题类别：frequency\n"
            f"用户问题：What to do for Mineral and Bone Disorder in Chronic Kidney Disease ?\n"
            "请基于医学常识与临床实践经验给出直接、重点突出的回答。回答应控制在200字以内。"
        ),
        (
            "你是一名资深的医学问答助手，擅长基于专业知识提供准确、可靠的解答。\n"
            f"问题类别：frequency\n"
            f"用户问题：What are the genetic changes related to Leydig cell hypoplasia ?\n"
            "请基于医学常识与临床实践经验给出直接、重点突出的回答。回答应控制在200字以内。"
        ),
        (
            "你是一名资深的医学问答助手，擅长基于专业知识提供准确、可靠的解答。\n"
            f"问题类别：frequency\n"
            f"用户问题：What are the treatments for Acanthamoeba - Granulomatous Amebic Encephalitis (GAE); Keratitis ?\n"
            "请基于医学常识与临床实践经验给出直接、重点突出的回答。回答应控制在200字以内。"
        ),
        (
            "你是一名资深的医学问答助手，擅长基于专业知识提供准确、可靠的解答。\n"
            f"问题类别：frequency\n"
            f"用户问题：What is (are) Chromosome 8p deletion ?\n"
            "请基于医学常识与临床实践经验给出直接、重点突出的回答。回答应控制在200字以内。"
        ),
        (
            "你是一名资深的医学问答助手，擅长基于专业知识提供准确、可靠的解答。\n"
            f"问题类别：frequency\n"
            f"用户问题：What are the genetic changes related to branchiootorenal/branchiootic syndrome ?\n"
            "请基于医学常识与临床实践经验给出直接、重点突出的回答。回答应控制在200字以内。"
        ),
        (
            "你是一名资深的医学问答助手，擅长基于专业知识提供准确、可靠的解答。\n"
            f"问题类别：frequency\n"
            f"用户问题：What are the genetic changes related to Hirschsprung disease ?\n"
            "请基于医学常识与临床实践经验给出直接、重点突出的回答。回答应控制在200字以内。"
        ),
        (
            "你是一名资深的医学问答助手，擅长基于专业知识提供准确、可靠的解答。\n"
            f"问题类别：frequency\n"
            f"用户问题：how can botulism be prevented?\n"
            "请基于医学常识与临床实践经验给出直接、重点突出的回答。回答应控制在200字以内。"
        ),
        (
            "你是一名资深的医学问答助手，擅长基于专业知识提供准确、可靠的解答。\n"
            f"问题类别：frequency\n"
            f"用户问题：What are the symptoms of Marburg hemorrhagic fever (Marburg HF) ?\n"
            "请基于医学常识与临床实践经验给出直接、重点突出的回答。回答应控制在200字以内。"
        ),
        (
            "你是一名资深的医学问答助手，擅长基于专业知识提供准确、可靠的解答。\n"
            f"问题类别：frequency\n"
            f"用户问题：Who is at risk for Omsk Hemorrhagic Fever (OHF)?\n"
            "请基于医学常识与临床实践经验给出直接、重点突出的回答。回答应控制在200字以内。"
        ),
        (
            "你是一名资深的医学问答助手，擅长基于专业知识提供准确、可靠的解答。\n"
            f"问题类别：frequency\n"
            f"用户问题：What are the treatments for Primary Lateral Sclerosis ?\n"
            "请基于医学常识与临床实践经验给出直接、重点突出的回答。回答应控制在200字以内。"
        ),
        (
            "你是一名资深的医学问答助手，擅长基于专业知识提供准确、可靠的解答。\n"
            f"问题类别：frequency\n"
            f"用户问题：what research (or clinical trials) is being done for Lambert-Eaton Myasthenic Syndrome ?\n"
            "请基于医学常识与临床实践经验给出直接、重点突出的回答。回答应控制在200字以内。"
        ),
        (
            "你是一名资深的医学问答助手，擅长基于专业知识提供准确、可靠的解答。\n"
            f"问题类别：frequency\n"
            f"用户问题：How many people are affected by Wolff-Parkinson-White syndrome ?\n"
            "请基于医学常识与临床实践经验给出直接、重点突出的回答。回答应控制在200字以内。"
        )
    ]
    
    return attack_variants[:population_size]