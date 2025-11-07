#!/usr/bin/env python
"""调试脚本：直接运行 location_wo_with_hooks.py 并查看完整输出"""
import subprocess
import sys

cmd = [
    sys.executable,
    "location_wo_with_hooks.py",
    "--user_input",
    "测试问题：1+1等于多少？\n选项：\nA. 1\nB. 2\nC. 3\nD. 4\nE. 5"
]

print("=" * 80)
print("运行命令:", " ".join(cmd))
print("=" * 80)

try:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
        env={"CUDA_VISIBLE_DEVICES": "3"}
    )

    print("\n[STDOUT]:")
    print("-" * 80)
    print(result.stdout)
    print("-" * 80)

    print("\n[STDERR]:")
    print("-" * 80)
    print(result.stderr)
    print("-" * 80)

    print(f"\n[返回码]: {result.returncode}")

    # 检查关键输出
    if "Pipeline生成的新文字:" in result.stdout:
        print("\n✓ 找到了生成文本标记")
    else:
        print("\n✗ 未找到生成文本标记")

    if "INTERMEDIATE_OUTPUTS_START" in result.stdout:
        print("✓ 找到了中间层输出标记")
    else:
        print("✗ 未找到中间层输出标记")

except subprocess.TimeoutExpired:
    print("\n[错误] 命令超时")
except Exception as e:
    print(f"\n[错误] {e}")
