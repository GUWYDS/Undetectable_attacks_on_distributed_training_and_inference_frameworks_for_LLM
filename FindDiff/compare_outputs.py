#!/usr/bin/env python3
"""
比较 location1.py 和 location2.py 中每个 API 调用的输出
"""
import pickle
import torch
import numpy as np
import os
from typing import Any, Tuple

def load_outputs(file1: str, file2: str) -> Tuple[dict, dict]:
    """加载两个输出文件"""
    if not os.path.exists(file1):
        raise FileNotFoundError(f"找不到文件: {file1}")
    if not os.path.exists(file2):
        raise FileNotFoundError(f"找不到文件: {file2}")
    
    with open(file1, 'rb') as f:
        outputs1 = pickle.load(f)
    
    with open(file2, 'rb') as f:
        outputs2 = pickle.load(f)
    
    return outputs1, outputs2

def compare_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor, tolerance: float = 1e-6) -> dict:
    """比较两个张量是否相等"""
    if tensor1.shape != tensor2.shape:
        return {"shape_equal": False, "msg": "形状不同"}
    mad = torch.mean(torch.abs(tensor1 - tensor2)).item()
    return {"shape_equal": True, "MAD": mad}

def compare_outputs(output1: Any, output2: Any, tolerance: float = 1e-6) -> dict:
    result = {
        "type_equal": type(output1) == type(output2),
        "comparison_results": []
    }
    
    if not result["type_equal"]:
        result["error"] = f"类型不匹配: {type(output1)} vs {type(output2)}"
        return result
    
    if isinstance(output1, torch.Tensor):
        result["comparison_results"] = compare_tensors(output1, output2, tolerance)
    elif isinstance(output1, (tuple, list)):
        for i, (item1, item2) in enumerate(zip(output1, output2)):
            if isinstance(item1, torch.Tensor) and isinstance(item2, torch.Tensor):
                item_result = compare_tensors(item1, item2, tolerance)
                result["comparison_results"].append({
                    "index": i,
                    "comparison": item_result
                })
            else:
                result["comparison_results"].append({
                    "index": i,
                    "equal": item1 == item2 if not isinstance(item1, torch.Tensor) else False,
                    "type1": type(item1).__name__,
                    "type2": type(item2).__name__
                })
    else:
        result["comparison_results"] = None
    
    return result

def main():
    print("开始比较 location1 和 location2 的输出...")
    
    # 加载输出数据
    outputs1, outputs2 = load_outputs("output_location1.pkl", "output_location2.pkl")
    
    print(f"Location1 输出数量: {len(outputs1)}")
    print(f"Location2 输出数量: {len(outputs2)}")
    print("-" * 80)

    common_keys = [key for key in outputs1.keys() if key in outputs2.keys()]
    print(f"共同的API调用: {len(common_keys)}")
    
    print("\n" + "=" * 80)
    print("详细比较结果:")
    print("=" * 80)
    
    identical_count = 0
    similar_count = 0
    different_count = 0
    
    # 记录到当前API为止见过的最大MAD值的数组
    max_mad_history = []
    e = 1e-7
    RL = []
    current_max_mad = 0.0
    
    for key in common_keys:
        output1 = outputs1[key]
        output2 = outputs2[key]
        
        print(f"\nAPI调用: {key}")
        print("-" * 40)
        
        comparison_result = compare_outputs(output1, output2)
        if isinstance(output1, torch.Tensor):
            comp = comparison_result["comparison_results"]
            if not comp["shape_equal"]:
                print("  形状不同")
                different_count += 1
                status = "✗ 形状不同"
            else:
                mad_value = comp['MAD']
                print(f"  MAD: {mad_value:.6e}")
                status = f"MAD={mad_value:.6e}"
                
            print(f"  状态: {status}")
            
        elif isinstance(output1, (tuple, list)):
            print(f"  类型: {type(output1).__name__}")
            print(f"  长度: {len(output1)}")
            print(f"  类型相等: {comparison_result['type_equal']}")
            
            layer_mad_values = []  # 收集当前API调用中所有tensor的MAD值
            
            for item_result in comparison_result["comparison_results"]:
                idx = item_result["index"]
                if "comparison" in item_result:
                    comp = item_result["comparison"]
                    if not comp["shape_equal"]:
                        print(f"    项目 {idx}: 形状不同")
                    else:
                        mad_value = comp['MAD']
                        print(f"    项目 {idx}: MAD={mad_value:.6e}")
                        layer_mad_values.append(mad_value)
                        
                else:
                    print(f"    项目 {idx}: 类型不同或不可比较")
        else:
            print(f"  类型: {type(output1).__name__}")
        
        # 记录当前API之前的最大MAD值（不包括当前API）
        max_mad_history.append(current_max_mad)
        print(f"  当前API之前的最大MAD: {current_max_mad:.6e}")
        
        # 计算当前API的MAD值用于RL计算
        current_api_mad = 0.0
        if isinstance(output1, torch.Tensor):
            comp = comparison_result["comparison_results"]
            if comp["shape_equal"]:
                current_api_mad = comp['MAD']
        elif isinstance(output1, (tuple, list)):
            # 对于tuple/list，找出所有MAD值中的最大值作为当前API的MAD
            for item_result in comparison_result["comparison_results"]:
                if "comparison" in item_result:
                    comp = item_result["comparison"]
                    if comp["shape_equal"]:
                        mad_value = comp['MAD']
                        current_api_mad = max(current_api_mad, mad_value)
        
        # 计算RL值
        if len(max_mad_history) > 0:  # 确保有历史数据
            prev_max_mad = max_mad_history[-1]  # 当前API之前的最大MAD
            RL.append((current_api_mad - prev_max_mad) / (prev_max_mad + e))
            print(f"  当前API MAD: {current_api_mad:.6e}")
            print(f"  RL值: {RL[-1]:.6e}")
        
        # 现在更新current_max_mad以包含当前API的MAD值（用于下一次记录）
        if current_api_mad > current_max_mad:
            current_max_mad = current_api_mad
    
    rl_90_percentile = np.percentile(RL, 95)
    print(f"所有RL的90分位数: {rl_90_percentile:.6e}")
    print(f"总RL数量: {len(RL)}")
    
    # 找出大于90分位数的API
    apis_above_90 = []
    for i, (api_key, rl_value) in enumerate(zip(common_keys, RL)):  # RL比common_keys少一个，从第二个API开始
        if rl_value > rl_90_percentile:
            apis_above_90.append((api_key, rl_value))
    
    print(f"\n大于90分位数的API数量: {len(apis_above_90)}")
    print("\n大于90分位数的API列表:")
    print("-" * 60)
    for api_key, rl_value in apis_above_90:
        print(f"API: {api_key}")
        print(f"RL值: {rl_value:.6e}")
        print("-" * 40)

if __name__ == "__main__":
    main()
