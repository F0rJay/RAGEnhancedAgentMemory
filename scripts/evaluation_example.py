#!/usr/bin/env python3
"""
评估示例

展示如何使用评估体系评估 RAG 系统质量。
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    """主函数"""
    print("=== 评估体系示例 ===\n")

    try:
        from src.evaluation.ragas_eval import RagasEvaluator
        from src.evaluation.needle_test import NeedleInHaystackTester
        
        # Ragas 评估示例
        print("1. Ragas 评估示例...")
        print("   注意: 需要安装 ragas 和 datasets")
        print("   运行: pip install ragas datasets")
        print()
        
        # Needle 测试示例
        print("2. Needle-in-a-Haystack 测试示例...")
        tester = NeedleInHaystackTester(
            token_estimate=10000,
            needle_positions=[0.0, 0.5, 1.0],
        )
        
        # 定义简单的 Agent 函数
        def agent_fn(haystack: str) -> str:
            """简单的 Agent 函数示例"""
            if "秘密启动密码是 9527" in haystack:
                return "9527"
            return "未找到关键信息"
        
        # 运行测试
        needle = "系统的秘密启动密码是 9527"
        print(f"   测试 Needle: {needle}")
        result = tester.run_test(agent_fn, needle, position=0.5)
        
        print(f"   结果: {'成功' if result.success else '失败'}")
        print(f"   准确度: {result.accuracy:.3f}")
        print()
        
        print("   更多评估示例请参考:")
        print("   - src/evaluation/evaluation_example.py")
        print("   - src/evaluation/ragas_eval.py")
        print("   - src/evaluation/needle_test.py")
        print()
        
    except ImportError as e:
        print(f"警告: 评估模块需要额外依赖")
        print(f"请运行: pip install ragas datasets")
        print(f"详细错误: {e}")
        print()
    
    print("=== 示例完成 ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
