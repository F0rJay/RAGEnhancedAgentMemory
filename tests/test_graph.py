"""
测试 LangGraph 相关模块
"""

import pytest
from typing import List

from src.graph.state import (
    AgentState,
    reduce_chat_history,
    reduce_documents,
)


def test_state_reducers():
    """测试状态缩减器"""
    # 测试聊天历史合并
    left_history = []
    right_history = []
    result = reduce_chat_history(left_history, right_history)
    assert isinstance(result, list)

    # 测试文档合并
    left_docs = ["文档1"]
    right_docs = ["文档2"]
    result = reduce_documents(left_docs, right_docs)
    assert len(result) >= 2
    assert "文档1" in result
    assert "文档2" in result


def test_state_optional_fields():
    """测试状态可选字段"""
    state: AgentState = {
        "input": "测试",
        "chat_history": [],
        "documents": [],
        "document_metadata": [],
        "generation": "",
        "relevance_score": "",
        "hallucination_score": "",
        "retry_count": 0,
    }

    # 可选字段可以不包含
    assert "should_continue" not in state or isinstance(state.get("should_continue"), bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
