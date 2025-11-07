import json

def find_first_token_diff(json1_data, json2_data):
    # 提取两个tokens列表和token_ids列表
    tokens1 = json1_data.get("tokens", [])
    tokens2 = json2_data.get("tokens", [])
    ids1 = json1_data.get("token_ids", [])
    ids2 = json2_data.get("token_ids", [])
    
    # 确保tokens和token_ids长度一致（避免数据异常）
    if len(tokens1) != len(ids1) or len(tokens2) != len(ids2):
        return -1, "JSON数据异常：tokens与token_ids长度不匹配"
    
    # 遍历所有索引，找到第一个差异
    max_len = min(len(tokens1), len(tokens2))  # 取较短长度，避免索引越界
    for idx in range(max_len):
        if tokens1[idx] != tokens2[idx] or ids1[idx] != ids2[idx]:
            # 返回差异详情：索引、两个token及对应ID
            detail = (
                f"第一个差异位置：索引{idx}\n"
                f"JSON1: token='{tokens1[idx]}' (ID={ids1[idx]})\n"
                f"JSON2: token='{tokens2[idx]}' (ID={ids2[idx]})\n"
                f"对应text片段：\n"
                f"JSON1: ...{json1_data['text'][max(0, idx-10):idx+20]}...\n"
                f"JSON2: ...{json2_data['text'][max(0, idx-10):idx+20]}..."
            )
            return idx, detail
    
    # 若前max_len个token一致，检查长度是否不同
    if len(tokens1) != len(tokens2):
        shorter = "JSON1" if len(tokens1) < len(tokens2) else "JSON2"
        longer = "JSON2" if len(tokens1) < len(tokens2) else "JSON1"
        detail = (
            f"前{max_len}个token完全一致，但长度不同：\n"
            f"{shorter}长度={min(len(tokens1), len(tokens2))}，{longer}长度={max(len(tokens1), len(tokens2))}\n"
            f"{longer}多出来的第一个token：{tokens1[max_len] if len(tokens1) > len(tokens2) else tokens2[max_len]}"
        )
        return max_len, detail
    
    # 完全无差异
    return -1, "两个JSON的tokens列表完全一致"

if __name__ == "__main__":
    # 1. 读取两个JSON文件（替换为你的文件路径）
    with open("result_wo.json", "r", encoding="utf-8") as f1:
        json1 = json.load(f1)
    with open("result_w.json", "r", encoding="utf-8") as f2:
        json2 = json.load(f2)
    
    # 2. 查找第一个差异
    diff_idx, diff_detail = find_first_token_diff(json1, json2)
    
    # 3. 输出结果
    print("="*50)
    if diff_idx == -1:
        print("结果：无差异")
    else:
        print(f"结果：第一个差异出现在索引{diff_idx}")
        print("详细对比：")
        print(diff_detail)
    print("="*50)