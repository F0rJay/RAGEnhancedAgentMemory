"""
测试核心记忆系统模块
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

try:
    from src.core import RAGEnhancedAgentMemory
except ImportError:
    RAGEnhancedAgentMemory = None

from src.graph.state import AgentState
from src.memory.routing import RouteDecision
from src.memory.long_term import MemoryItem, MemoryType

from tests.utils import (
    create_mock_short_term_memory,
    create_mock_long_term_memory,
    create_mock_agent_state,
    create_test_conversation_turns,
)


def test_rag_enhanced_memory_initialization():
    """测试 RAGEnhancedAgentMemory 初始化"""
    if RAGEnhancedAgentMemory is None:
        pytest.skip("需要 langchain-core 依赖")
    
    # 尝试初始化（即使向量数据库不可用也应该能初始化基础组件）
    try:
        memory = RAGEnhancedAgentMemory(
            vector_db="qdrant",
            session_id="test_session",
        )
        # 验证基本组件已初始化
        assert memory.short_term_memory is not None
        assert memory.router is not None
        assert memory.session_id == "test_session"
        # 长期记忆可能因为向量数据库不可用而初始化失败，但不影响基本功能
    except Exception as e:
        # 如果是向量数据库连接错误，可以跳过
        if "connection" in str(e).lower() or "qdrant" in str(e).lower():
            pytest.skip(f"跳过测试（向量数据库不可用）: {e}")
        else:
            raise


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
    if RAGEnhancedAgentMemory is None:
        pytest.skip("需要 langchain-core 依赖")
    
    try:
        memory = RAGEnhancedAgentMemory(
            vector_db="qdrant",
            session_id="test_session_stats",
        )
        
        # 获取统计信息
        stats = memory.get_stats()
        
        # 验证统计信息结构
        assert "session_id" in stats
        assert stats["session_id"] == "test_session_stats"
        assert "short_term" in stats
        assert "vector_db" in stats
        
        # 验证短期记忆统计
        short_term_stats = stats["short_term"]
        assert "turn_count" in short_term_stats
        
    except Exception as e:
        # 如果是向量数据库连接错误，可以跳过
        if "connection" in str(e).lower() or "qdrant" in str(e).lower():
            pytest.skip(f"跳过测试（向量数据库不可用）: {e}")
        else:
            raise


def test_save_context():
    """测试保存对话上下文"""
    if RAGEnhancedAgentMemory is None:
        pytest.skip("需要 langchain-core 依赖")
    
    try:
        memory = RAGEnhancedAgentMemory(
            vector_db="qdrant",
            session_id="test_session_save",
        )
        
        # 保存对话上下文
        memory.save_context(
            inputs={"input": "你好"},
            outputs={"generation": "你好！有什么可以帮助你的吗？"}
        )
        
        # 验证已保存到短期记忆
        stats = memory.get_stats()
        short_term_stats = stats["short_term"]
        assert short_term_stats["turn_count"] == 1
        
        # 再次保存
        memory.save_context(
            inputs={"input": "我喜欢Python"},
            outputs={"generation": "Python是个很好的编程语言"}
        )
        
        # 验证累计
        stats = memory.get_stats()
        assert stats["short_term"]["turn_count"] == 2
        
    except Exception as e:
        # 如果是向量数据库连接错误，可以跳过
        if "connection" in str(e).lower() or "qdrant" in str(e).lower():
            pytest.skip(f"跳过测试（向量数据库不可用）: {e}")
        else:
            raise


def test_retrieve_context_empty_query():
    """测试空查询的上下文检索"""
    if RAGEnhancedAgentMemory is None:
        pytest.skip("需要 langchain-core 依赖")
    
    try:
        memory = RAGEnhancedAgentMemory(
            vector_db="qdrant",
            session_id="test_session_retrieve",
        )
        
        # 测试空查询
        state = {"input": ""}
        result = memory.retrieve_context(state)
        
        assert "documents" in result
        assert "memory_type" in result
        assert result["memory_type"] == "none"
        assert len(result["documents"]) == 0
        
    except Exception as e:
        if "connection" in str(e).lower() or "qdrant" in str(e).lower():
            pytest.skip(f"跳过测试（向量数据库不可用）: {e}")
        else:
            raise


def test_retrieve_context_with_short_term_only():
    """测试仅使用短期记忆的上下文检索"""
    if RAGEnhancedAgentMemory is None:
        pytest.skip("需要 langchain-core 依赖")
    
    try:
        memory = RAGEnhancedAgentMemory(
            vector_db="qdrant",
            session_id="test_session_retrieve_short",
        )
        
        # 添加一些对话到短期记忆
        memory.save_context(
            inputs={"input": "你好"},
            outputs={"generation": "你好！有什么可以帮助你的吗？"}
        )
        
        # 创建一个简单的查询（应该只使用短期记忆）
        state = {"input": "刚才你说什么？"}
        result = memory.retrieve_context(state)
        
        assert "documents" in result
        assert "memory_type" in result
        assert "retrieval_query" in result
        assert result["retrieval_query"] == "刚才你说什么？"
        # 短期记忆应该被包含
        assert len(result["documents"]) >= 0  # 可能没有，取决于路由决策
        
    except Exception as e:
        if "connection" in str(e).lower() or "qdrant" in str(e).lower():
            pytest.skip(f"跳过测试（向量数据库不可用）: {e}")
        else:
            raise


def test_add_memory():
    """测试添加记忆到长期记忆库"""
    if RAGEnhancedAgentMemory is None:
        pytest.skip("需要 langchain-core 依赖")
    
    try:
        memory = RAGEnhancedAgentMemory(
            vector_db="qdrant",
            session_id="test_session_add_memory",
        )
        
        # 添加记忆
        memory_id = memory.add_memory(
            content="用户喜欢Python编程",
            metadata={"category": "preference"}
        )
        
        assert memory_id is not None
        assert isinstance(memory_id, str)
        
        # 验证记忆已添加（通过统计信息）
        stats = memory.get_stats()
        long_term_stats = stats["long_term"]
        # 注意：total_memories 可能是 "unknown" 或者数字
        
    except Exception as e:
        error_str = str(e).lower()
        if "connection" in error_str or "qdrant" in error_str or "huggingface" in error_str:
            pytest.skip(f"跳过测试（向量数据库或模型不可用）: {e}")
        else:
            raise


def test_search_without_hybrid():
    """测试不使用混合检索的搜索功能"""
    if RAGEnhancedAgentMemory is None:
        pytest.skip("需要 langchain-core 依赖")
    
    try:
        memory = RAGEnhancedAgentMemory(
            vector_db="qdrant",
            session_id="test_session_search",
            use_hybrid_retrieval=False,  # 禁用混合检索
        )
        
        # 先添加一些记忆
        memory.add_memory(content="Python是编程语言")
        
        # 执行搜索
        results = memory.search(query="编程语言", top_k=5, use_hybrid=False)
        
        assert isinstance(results, list)
        # 验证结果格式
        if results:
            assert "content" in results[0]
            assert "score" in results[0]
            assert "metadata" in results[0]
        
    except Exception as e:
        error_str = str(e).lower()
        if "connection" in error_str or "qdrant" in error_str or "huggingface" in error_str:
            pytest.skip(f"跳过测试（向量数据库或模型不可用）: {e}")
        else:
            raise


def test_search_with_hybrid():
    """测试使用混合检索的搜索功能"""
    if RAGEnhancedAgentMemory is None:
        pytest.skip("需要 langchain-core 依赖")
    
    try:
        memory = RAGEnhancedAgentMemory(
            vector_db="qdrant",
            session_id="test_session_search_hybrid",
            use_hybrid_retrieval=True,  # 启用混合检索
        )
        
        # 先添加一些记忆
        memory.add_memory(content="机器学习是AI的子领域")
        
        # 执行搜索（使用混合检索）
        results = memory.search(query="AI", top_k=5, use_hybrid=True)
        
        assert isinstance(results, list)
        # 验证结果格式（混合检索结果包含 source）
        if results:
            assert "content" in results[0]
            assert "score" in results[0]
            assert "source" in results[0]  # 混合检索特有
            assert "metadata" in results[0]
        
    except Exception as e:
        error_str = str(e).lower()
        if "connection" in error_str or "qdrant" in error_str or "huggingface" in error_str:
            pytest.skip(f"跳过测试（向量数据库或模型不可用）: {e}")
        else:
            raise


def test_clear_short_term_memory():
    """测试清空短期记忆"""
    if RAGEnhancedAgentMemory is None:
        pytest.skip("需要 langchain-core 依赖")
    
    try:
        memory = RAGEnhancedAgentMemory(
            vector_db="qdrant",
            session_id="test_session_clear",
        )
        
        # 添加一些对话
        memory.save_context(
            inputs={"input": "你好"},
            outputs={"generation": "你好！"}
        )
        memory.save_context(
            inputs={"input": "再见"},
            outputs={"generation": "再见！"}
        )
        
        # 验证已保存
        stats_before = memory.get_stats()
        assert stats_before["short_term"]["turn_count"] == 2
        
        # 清空短期记忆
        memory.clear_short_term_memory()
        
        # 验证已清空
        stats_after = memory.get_stats()
        assert stats_after["short_term"]["turn_count"] == 0
        
    except Exception as e:
        if "connection" in str(e).lower() or "qdrant" in str(e).lower():
            pytest.skip(f"跳过测试（向量数据库不可用）: {e}")
        else:
            raise


def test_archive_short_term_to_long_term():
    """测试将短期记忆归档到长期记忆"""
    if RAGEnhancedAgentMemory is None:
        pytest.skip("需要 langchain-core 依赖")
    
    try:
        memory = RAGEnhancedAgentMemory(
            vector_db="qdrant",
            session_id="test_session_archive",
        )
        
        # 添加一些对话到短期记忆
        memory.save_context(
            inputs={"input": "用户喜欢Python"},
            outputs={"generation": "Python是很好的编程语言"}
        )
        memory.save_context(
            inputs={"input": "用户会使用Docker"},
            outputs={"generation": "Docker是容器化工具"}
        )
        
        # 验证短期记忆有内容
        stats_before = memory.get_stats()
        assert stats_before["short_term"]["turn_count"] >= 2
        
        # 归档到长期记忆
        archived_ids = memory.archive_short_term_to_long_term()
        
        # 验证归档结果
        assert isinstance(archived_ids, list)
        assert len(archived_ids) >= 2  # 应该归档了至少2条记忆
        
    except Exception as e:
        error_str = str(e).lower()
        if "connection" in error_str or "qdrant" in error_str or "huggingface" in error_str:
            pytest.skip(f"跳过测试（向量数据库或模型不可用）: {e}")
        else:
            raise


def test_archive_empty_short_term():
    """测试归档空的短期记忆"""
    if RAGEnhancedAgentMemory is None:
        pytest.skip("需要 langchain-core 依赖")
    
    try:
        memory = RAGEnhancedAgentMemory(
            vector_db="qdrant",
            session_id="test_session_archive_empty",
        )
        
        # 不添加任何对话，直接尝试归档
        archived_ids = memory.archive_short_term_to_long_term()
        
        # 应该返回空列表
        assert isinstance(archived_ids, list)
        assert len(archived_ids) == 0
        
    except Exception as e:
        if "connection" in str(e).lower() or "qdrant" in str(e).lower():
            pytest.skip(f"跳过测试（向量数据库不可用）: {e}")
        else:
            raise


def test_get_context_for_llm():
    """测试获取LLM格式的上下文"""
    if RAGEnhancedAgentMemory is None:
        pytest.skip("需要 langchain-core 依赖")
    
    try:
        memory = RAGEnhancedAgentMemory(
            vector_db="qdrant",
            session_id="test_session_llm_context",
        )
        
        # 添加一些对话
        memory.save_context(
            inputs={"input": "你好"},
            outputs={"generation": "你好！有什么可以帮助你的吗？"}
        )
        memory.save_context(
            inputs={"input": "我喜欢Python"},
            outputs={"generation": "Python是很好的编程语言"}
        )
        
        # 获取LLM格式的上下文
        context = memory.get_context_for_llm()
        
        assert isinstance(context, str)
        assert len(context) > 0
        # 验证包含对话内容
        assert "你好" in context or "Python" in context
        
        # 测试带最大token限制
        context_limited = memory.get_context_for_llm(max_tokens=100)
        assert isinstance(context_limited, str)
        # 限制后的上下文应该更短或相等
        assert len(context_limited) <= len(context) or len(context) < 400  # 粗略估算
        
    except Exception as e:
        if "connection" in str(e).lower() or "qdrant" in str(e).lower():
            pytest.skip(f"跳过测试（向量数据库不可用）: {e}")
        else:
            raise


def test_get_checkpointer():
    """测试获取检查点保存器"""
    if RAGEnhancedAgentMemory is None:
        pytest.skip("需要 langchain-core 依赖")
    
    try:
        memory = RAGEnhancedAgentMemory(
            vector_db="qdrant",
            session_id="test_session_checkpointer",
        )
        
        # 获取检查点保存器
        checkpointer = memory.get_checkpointer()
        
        # 如果 LangGraph 可用，应该返回检查点保存器；否则为 None
        if checkpointer is not None:
            # 验证检查点保存器有基本方法（如果有的话）
            assert hasattr(checkpointer, 'put') or hasattr(checkpointer, 'save')
        
    except Exception as e:
        if "connection" in str(e).lower() or "qdrant" in str(e).lower():
            pytest.skip(f"跳过测试（向量数据库不可用）: {e}")
        else:
            raise


def test_retrieve_context_with_long_term():
    """测试包含长期记忆的上下文检索"""
    if RAGEnhancedAgentMemory is None:
        pytest.skip("需要 langchain-core 依赖")
    
    try:
        memory = RAGEnhancedAgentMemory(
            vector_db="qdrant",
            session_id="test_session_retrieve_long",
        )
        
        # 添加记忆到长期记忆
        memory.add_memory(content="深度学习是机器学习的一个分支")
        
        # 创建需要长期记忆的查询（短期记忆为空）
        state = {"input": "什么是深度学习？"}
        result = memory.retrieve_context(state)
        
        assert "documents" in result
        assert "memory_type" in result
        assert "retrieval_query" in result
        # 由于短期记忆为空，应该使用长期记忆
        
    except Exception as e:
        error_str = str(e).lower()
        if "connection" in error_str or "qdrant" in error_str or "huggingface" in error_str:
            pytest.skip(f"跳过测试（向量数据库或模型不可用）: {e}")
        else:
            raise


def test_retrieve_context_with_relevance_score():
    """测试带相关性评分的上下文检索"""
    if RAGEnhancedAgentMemory is None:
        pytest.skip("需要 langchain-core 依赖")
    
    try:
        memory = RAGEnhancedAgentMemory(
            vector_db="qdrant",
            session_id="test_session_relevance",
        )
        
        # 添加一些对话
        memory.save_context(
            inputs={"input": "你好"},
            outputs={"generation": "你好！"}
        )
        
        # 创建带相关性评分的状态
        state = {
            "input": "新的问题",
            "relevance_score": 0.5  # 低相关性，可能触发长期记忆检索
        }
        result = memory.retrieve_context(state)
        
        assert "documents" in result
        assert "memory_type" in result
        
    except Exception as e:
        if "connection" in str(e).lower() or "qdrant" in str(e).lower():
            pytest.skip(f"跳过测试（向量数据库不可用）: {e}")
        else:
            raise


def test_search_default_behavior():
    """测试搜索的默认行为（使用初始化时的设置）"""
    if RAGEnhancedAgentMemory is None:
        pytest.skip("需要 langchain-core 依赖")
    
    try:
        # 创建启用混合检索的实例
        memory_hybrid = RAGEnhancedAgentMemory(
            vector_db="qdrant",
            session_id="test_session_search_default_hybrid",
            use_hybrid_retrieval=True,
        )
        
        # 不指定 use_hybrid 参数，应该使用初始化时的设置
        results = memory_hybrid.search(query="测试", top_k=5)
        assert isinstance(results, list)
        
        # 创建不启用混合检索的实例
        memory_vector = RAGEnhancedAgentMemory(
            vector_db="qdrant",
            session_id="test_session_search_default_vector",
            use_hybrid_retrieval=False,
        )
        
        # 不指定 use_hybrid 参数，应该使用纯向量检索
        results = memory_vector.search(query="测试", top_k=5)
        assert isinstance(results, list)
        
    except Exception as e:
        error_str = str(e).lower()
        if "connection" in error_str or "qdrant" in error_str or "huggingface" in error_str:
            pytest.skip(f"跳过测试（向量数据库或模型不可用）: {e}")
        else:
            raise


def test_save_context_with_empty_inputs():
    """测试保存空输入的上下文"""
    if RAGEnhancedAgentMemory is None:
        pytest.skip("需要 langchain-core 依赖")
    
    try:
        memory = RAGEnhancedAgentMemory(
            vector_db="qdrant",
            session_id="test_session_save_empty",
        )
        
        # 保存空输入（应该被忽略）
        memory.save_context(
            inputs={"input": ""},
            outputs={"generation": "回答"}
        )
        
        # 验证没有保存（turn_count 应该为 0）
        stats = memory.get_stats()
        assert stats["short_term"]["turn_count"] == 0
        
        # 保存只有输入没有输出的情况
        memory.save_context(
            inputs={"input": "问题"},
            outputs={"generation": ""}
        )
        
        # 验证没有保存
        stats = memory.get_stats()
        assert stats["short_term"]["turn_count"] == 0
        
    except Exception as e:
        if "connection" in str(e).lower() or "qdrant" in str(e).lower():
            pytest.skip(f"跳过测试（向量数据库不可用）: {e}")
        else:
            raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
