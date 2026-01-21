#!/usr/bin/env python3
"""
基础使用示例

展示如何使用 RAGEnhancedAgentMemory 进行基本的记忆管理。
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core import RAGEnhancedAgentMemory


def main():
    """主函数"""
    print("=== RAG 增强型 Agent 记忆系统 - 基础示例 ===\n")

    # 初始化记忆系统
    print("1. 初始化记忆系统...")
    memory = RAGEnhancedAgentMemory(
        vector_db="qdrant",
        embedding_model="BAAI/bge-large-en-v1.5",
        session_id="demo_session_001",
    )
    print("   ✓ 初始化完成\n")

    # 添加记忆
    print("2. 添加记忆到长期记忆库...")
    memory_id = memory.add_memory(
        content="用户喜欢使用 Python 编程语言进行开发",
        metadata={"category": "preference", "topic": "programming"},
    )
    print(f"   ✓ 记忆已添加，ID: {memory_id[:8]}...\n")

    # 搜索记忆
    print("3. 搜索记忆...")
    results = memory.search("用户的技术偏好", top_k=3)
    print(f"   ✓ 检索到 {len(results)} 条相关记忆")
    for i, result in enumerate(results, 1):
        print(f"   {i}. {result['content'][:60]}... (评分: {result['score']:.2f})")
    print()

    # 添加对话到短期记忆
    print("4. 添加对话到短期记忆...")
    memory.save_context(
        inputs={"input": "我喜欢用 Python 做数据分析"},
        outputs={"generation": "Python 在数据科学领域非常流行，有很多强大的库如 pandas 和 numpy。"},
    )
    print("   ✓ 对话已保存到短期记忆\n")

    # 获取统计信息
    print("5. 获取系统统计信息...")
    stats = memory.get_stats()
    print(f"   会话ID: {stats['session_id']}")
    print(f"   短期记忆轮数: {stats['short_term']['turn_count']}")
    print(f"   向量数据库: {stats['vector_db']}")
    print()

    print("=== 示例完成 ===")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"错误: 缺少依赖。请运行: pip install -r requirements.txt")
        print(f"详细错误: {e}")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
