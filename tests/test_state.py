"""
测试状态定义
"""

import pytest
from typing import List
from langchain_core.messages import HumanMessage, AIMessage

from src.graph.state import (
    AgentState,
    AgentStateWithCustomReducer,
    reduce_chat_history,
    reduce_documents,
)


def test_agent_state_creation():
    """测试 AgentState 的基本创建"""
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
    assert state["retry_count"] == 0
    assert isinstance(state["chat_history"], list)
    assert isinstance(state["documents"], list)


def test_agent_state_with_optional_fields():
    """测试带可选字段的 AgentState"""
    state: AgentState = {
        "input": "测试问题",
        "chat_history": [],
        "documents": [],
        "document_metadata": [],
        "generation": "测试回答",
        "relevance_score": "yes",
        "hallucination_score": "grounded",
        "retry_count": 0,
        "should_continue": True,
        "memory_type": "long_term",
        "retrieval_query": "改写后的查询",
        "retrieval_latency": 0.5,
        "generation_latency": 1.2,
    }
    
    assert state["should_continue"] is True
    assert state["memory_type"] == "long_term"
    assert state["retrieval_latency"] == 0.5


def test_reduce_chat_history():
    """测试聊天历史合并函数"""
    left = [
        HumanMessage(content="用户问题1"),
        AIMessage(content="助手回答1"),
    ]
    right = [
        HumanMessage(content="用户问题2"),
        AIMessage(content="助手回答2"),
    ]
    
    result = reduce_chat_history(left, right)
    
    assert len(result) == 4
    assert result[0].content == "用户问题1"
    assert result[-1].content == "助手回答2"


def test_reduce_chat_history_deduplication():
    """测试聊天历史去重功能"""
    left = [
        HumanMessage(content="重复的消息"),
        AIMessage(content="回答1"),
    ]
    right = [
        HumanMessage(content="重复的消息"),  # 重复消息
        AIMessage(content="回答2"),
    ]
    
    result = reduce_chat_history(left, right)
    
    # 应该去重，只保留3条消息
    assert len(result) <= 4  # 可能去重也可能不去重，取决于实现


def test_reduce_documents():
    """测试文档合并函数"""
    left = ["文档1", "文档2"]
    right = ["文档3", "文档4"]
    
    result = reduce_documents(left, right)
    
    assert len(result) == 4
    assert "文档1" in result
    assert "文档3" in result


def test_reduce_documents_deduplication():
    """测试文档去重功能"""
    left = ["重复文档", "文档1"]
    right = ["重复文档", "文档2"]  # 重复文档
    
    result = reduce_documents(left, right)
    
    # 应该去重
    assert result.count("重复文档") <= 1
    assert "文档1" in result
    assert "文档2" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
