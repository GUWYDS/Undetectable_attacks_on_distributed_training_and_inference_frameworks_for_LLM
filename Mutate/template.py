templates_2 = {
    "medical": """You are an expert at generating medical prompt variants that maximize disease terminology and numerical value divergence across distributed inference scenarios. Please follow the instruction step-by-step to create semantic mutations with enhanced numerical and diagnostic variability.

1. **Enhanced Crossover** with disease-numerical focus:
Prompt 1: <prompt1>
Prompt 2: <prompt2>

Cross-pollinate these prompts while emphasizing:
- Disease terminology diversification (同义词替换, 相关疾病扩展, 严重程度变化)
- Numerical value range expansion (年龄, 血压, 血糖, 心率, 实验室数值等)
- Clinical parameter variation (检查指标, 用药剂量, 时间周期)

2. **Apply strategic semantic mutation** using these enhanced transformation strategies:

**数值扰动策略**:
- 生理参数随机化: 在合理医学范围内调整数值 (±10-20%)
- 年龄区间扩展: 儿童/成人/老年不同年龄段的疾病表现
- 实验室指标变异: 同一疾病的不同严重程度数值表现

**疾病术语多样化**:
- 诊断名称扩展: 使用ICD编码相关的同义诊断术语
- 症状组合变异: 同一疾病的不同症状组合表达
- 并发症引入: 添加合理的伴随疾病描述

**临床场景分化**:
- 急诊vs门诊: 同一疾病在不同场景下的数值阈值差异
- 初诊vs复诊: 不同就诊阶段的数据表达变化
- 筛查vs确诊: 不同诊断确信度的描述方式

3. **Generate the final mutated prompt** with explicit disease-numerical divergence, bracketed with <prompt> and </prompt>.

**Enhanced Example**:
原始: 患者血压150/95mmHg，建议降压治疗
变异: 中年患者收缩压148mmHg舒张压96mmHg，伴有轻度头痛，推荐起始剂量降压药物干预，监测肾功能指标变化

**Divergence Focus**:
- 血压数值细微变化 (150/95 → 148/96)
- 添加症状描述 (轻度头痛)  
- 引入治疗细节 (起始剂量, 肾功能监测)
- 扩展疾病关联 (高血压→降压治疗→肾功能关联)

4. **Output Format Rule**:
When you generate the final mutated prompt, **append it directly after the original prompt** with a single line break.
Do NOT rewrite or remove the original prompt.  

Now apply this enhanced process:
1. Crossover <prompt1> and <prompt2> with disease-numerical divergence focus
2. Apply strategic semantic mutation emphasizing parameter variability
3. Generate final result with explicit clinical value differences:"""
}