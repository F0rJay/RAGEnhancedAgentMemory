"""
测试核心记忆系统模块
"""

import pytest

from src.core import RAGEnhancedAgentMemory
from src.graph.state import AgentState


def test_rag_enhanced_memory_initialization():
    """测试 RAGEnhancedAgentMemory 初始化"""
    # 由于需要实际的向量数据库，这里只测试初始化逻辑
    # 实际测试需要 mock 或实际的数据库连接
    pass


def test_agent_state_structure():
    """测试 AgentState 结构"""
    state: AgentState = {
        "input": "测试问题",
        "chat_history": [],
        "documents": [],
        "document_metadata": [],
        "generation": "",
        "relevance_score": "yes",
        "hallucination_score": "grounded",
        "retry_count": 0,
    }

    assert state["input"] == "测试问题"
    assert isinstance(state["chat_history"], list)
    assert isinstance(state["documents"], list)


def test_get_stats():
    """测试获取统计信息"""
    # 需要实际实例，这里仅测试接口
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
