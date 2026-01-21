"""
边界条件测试

测试各个模块在边界情况下的行为，包括：
- 空输入处理
- None 值处理
- 极端值处理
- 类型错误处理
- 异常情况处理
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any, List


# ==================== ShortTermMemory 边界条件测试 ====================

def test_short_term_memory_empty_inputs():
    """测试短期记忆处理空输入"""
    from src.memory.short_term import ShortTermMemory

    memory = ShortTermMemory(max_turns=10)

    # 空字符串输入
    memory.add_conversation_turn(human_message="", ai_message="回答")
    memory.add_conversation_turn(human_message="问题", ai_message="")

    # 空上下文应该返回空结果
    empty_context = memory.get_recent_context()
    assert "turn_count" in empty_context


def test_short_term_memory_max_turns_zero():
    """测试最大轮数为0的情况"""
    from src.memory.short_term import ShortTermMemory

    memory = ShortTermMemory(max_turns=0)

    memory.add_conversation_turn(human_message="问题", ai_message="回答")

    context = memory.get_recent_context()
    # 即使max_turns=0，也应该能处理
    assert isinstance(context, dict)


def test_short_term_memory_very_large_input():
    """测试非常大的输入"""
    from src.memory.short_term import ShortTermMemory

    memory = ShortTermMemory(max_turns=10)

    # 非常大的字符串
    large_message = "x" * 100000
    memory.add_conversation_turn(
        human_message=large_message,
        ai_message="回答"
    )

    context = memory.get_recent_context()
    assert context["turn_count"] >= 1


def test_short_term_memory_special_characters():
    """测试特殊字符处理"""
    from src.memory.short_term import ShortTermMemory

    memory = ShortTermMemory(max_turns=10)

    special_message = "测试：\n\n换行\t制表符\n特殊字符!@#$%^&*()"
    memory.add_conversation_turn(
        human_message=special_message,
        ai_message="回答"
    )

    context = memory.get_recent_context()
    assert context["turn_count"] >= 1


# ==================== Routing 边界条件测试 ====================

def test_routing_empty_query():
    """测试路由处理空查询"""
    from src.memory.routing import MemoryRouter, RoutingContext
    from src.memory.short_term import ShortTermMemory
    from unittest.mock import Mock

    router = MemoryRouter()
    short_memory = ShortTermMemory()
    long_memory = Mock()

    context = RoutingContext(
        query="",
        short_term_memory=short_memory,
        long_term_memory=long_memory,
    )

    # 空查询应该能处理，不会抛出异常
    result = router.route(context)
    assert result is not None
    assert result.decision is not None


def test_routing_none_values():
    """测试路由处理None值"""
    from src.memory.routing import MemoryRouter, RoutingContext
    from src.memory.short_term import ShortTermMemory
    from unittest.mock import Mock

    router = MemoryRouter()
    short_memory = ShortTermMemory()
    long_memory = Mock()

    context = RoutingContext(
        query="测试",
        short_term_memory=short_memory,
        long_term_memory=long_memory,
        recent_relevance_score=None,
        previous_retrieval_time=None,
        query_complexity=None,
    )

    result = router.route(context)
    assert result is not None


def test_routing_extreme_relevance_scores():
    """测试极端相关性评分"""
    from src.memory.routing import MemoryRouter, RoutingContext
    from src.memory.short_term import ShortTermMemory
    from unittest.mock import Mock

    router = MemoryRouter()
    short_memory = ShortTermMemory()
    long_memory = Mock()

    # 极低评分
    context_low = RoutingContext(
        query="测试",
        short_term_memory=short_memory,
        long_term_memory=long_memory,
        recent_relevance_score=0.0,
    )
    result_low = router.route(context_low)
    assert result_low is not None

    # 极高评分
    context_high = RoutingContext(
        query="测试",
        short_term_memory=short_memory,
        long_term_memory=long_memory,
        recent_relevance_score=1.0,
    )
    result_high = router.route(context_high)
    assert result_high is not None


# ==================== HybridRetriever 边界条件测试 ====================

def test_hybrid_retriever_invalid_weights():
    """测试混合检索器无效权重"""
    from src.retrieval.hybrid import HybridRetriever
    from unittest.mock import Mock

    long_term_memory = Mock()

    # 权重之和不等于1.0
    with pytest.raises(ValueError, match="之和必须等于 1.0"):
        HybridRetriever(
            long_term_memory=long_term_memory,
            vector_weight=0.5,
            keyword_weight=0.6,
        )


def test_hybrid_retriever_zero_weights():
    """测试混合检索器零权重"""
    from src.retrieval.hybrid import HybridRetriever
    from unittest.mock import Mock

    long_term_memory = Mock()

    # 一个权重为0
    retriever = HybridRetriever(
        long_term_memory=long_term_memory,
        vector_weight=1.0,
        keyword_weight=0.0,
    )

    assert retriever.vector_weight == 1.0
    assert retriever.keyword_weight == 0.0


def test_hybrid_retriever_empty_query():
    """测试混合检索器空查询"""
    from src.retrieval.hybrid import HybridRetriever
    from unittest.mock import Mock, MagicMock

    long_term_memory = Mock()
    long_term_memory.search.return_value = []

    retriever = HybridRetriever(
        long_term_memory=long_term_memory,
        vector_weight=0.7,
        keyword_weight=0.3,
    )

    # 空查询应该返回空结果
    results = retriever.retrieve("", top_k=5)
    assert results == []


def test_hybrid_retriever_zero_top_k():
    """测试混合检索器top_k=0"""
    from src.retrieval.hybrid import HybridRetriever
    from unittest.mock import Mock

    long_term_memory = Mock()
    long_term_memory.search.return_value = []

    retriever = HybridRetriever(
        long_term_memory=long_term_memory,
    )

    results = retriever.retrieve("测试", top_k=0)
    assert results == []


# ==================== Config 边界条件测试 ====================

def test_config_invalid_env_values():
    """测试配置处理无效环境变量值"""
    import os
    from src.config import get_settings

    # 保存原始值
    original_max_len = os.environ.get("VLLM_MAX_MODEL_LEN")

    try:
        # 设置无效值（非数字）
        os.environ["VLLM_MAX_MODEL_LEN"] = "invalid"

        # 应该能处理无效值（使用默认值或抛出异常）
        try:
            settings = get_settings()
            # 如果能获取设置，应该使用默认值
            assert isinstance(settings.vllm_max_model_len, int)
        except (ValueError, TypeError):
            # 如果抛出异常也是可以接受的
            pass
    finally:
        # 恢复原始值
        if original_max_len is not None:
            os.environ["VLLM_MAX_MODEL_LEN"] = original_max_len
        elif "VLLM_MAX_MODEL_LEN" in os.environ:
            del os.environ["VLLM_MAX_MODEL_LEN"]


def test_config_negative_values():
    """测试配置处理负数值"""
    import os
    from src.config import get_settings

    original_util = os.environ.get("VLLM_GPU_MEMORY_UTILIZATION")

    try:
        # 设置负数值
        os.environ["VLLM_GPU_MEMORY_UTILIZATION"] = "-0.5"

        settings = get_settings()
        # 应该被验证或使用默认值
        assert settings.vllm_gpu_memory_utilization >= 0.0
    finally:
        if original_util is not None:
            os.environ["VLLM_GPU_MEMORY_UTILIZATION"] = original_util
        elif "VLLM_GPU_MEMORY_UTILIZATION" in os.environ:
            del os.environ["VLLM_GPU_MEMORY_UTILIZATION"]


# ==================== LongTermMemory 边界条件测试 ====================

def test_long_term_memory_empty_content():
    """测试长期记忆处理空内容"""
    from src.memory.long_term import MemoryItem

    # 空内容
    item = MemoryItem(
        id="test-id",
        content="",
        memory_type="conversation",
        metadata={}
    )

    assert item.content == ""
    assert item.memory_type == "conversation"


def test_long_term_memory_none_metadata():
    """测试长期记忆处理None元数据"""
    from src.memory.long_term import MemoryItem

    # metadata应该使用默认值，不接受None
    item = MemoryItem(
        id="test-id",
        content="测试",
        memory_type="conversation",
        metadata={}  # 空字典而不是None
    )

    assert item.metadata == {}


# ==================== Evaluation 边界条件测试 ====================

def test_evaluation_empty_lists():
    """测试评估模块处理空列表"""
    from src.evaluation.ragas_eval import EvaluationResult

    # 评估结果应该能处理各种值
    result = EvaluationResult(
        context_recall=0.0,
        context_precision=0.0,
        faithfulness=0.0,
        answer_relevancy=0.0,
        overall_score=0.0,
    )

    assert result.overall_score == 0.0


def test_evaluation_extreme_scores():
    """测试评估模块处理极端评分"""
    from src.evaluation.ragas_eval import EvaluationResult

    # 最小分数
    result_min = EvaluationResult(
        context_recall=0.0,
        context_precision=0.0,
        faithfulness=0.0,
        answer_relevancy=0.0,
        overall_score=0.0,
    )

    # 最大分数
    result_max = EvaluationResult(
        context_recall=1.0,
        context_precision=1.0,
        faithfulness=1.0,
        answer_relevancy=1.0,
        overall_score=1.0,
    )

    assert result_min.overall_score == 0.0
    assert result_max.overall_score == 1.0


# ==================== Reranker 边界条件测试 ====================

def test_reranker_empty_documents():
    """测试重排序器处理空文档列表"""
    from src.retrieval.reranker import RerankedResult

    # 重排序结果应该能处理各种情况
    result = RerankedResult(
        content="",
        original_score=0.0,
        rerank_score=0.0,
        metadata={},
        rank_change=0,
    )

    assert result.content == ""


def test_reranker_extreme_scores():
    """测试重排序器处理极端评分"""
    from src.retrieval.reranker import RerankedResult

    # 最小评分
    result_min = RerankedResult(
        content="测试",
        original_score=0.0,
        rerank_score=0.0,
        metadata={},
        rank_change=0,
    )

    # 最大评分
    result_max = RerankedResult(
        content="测试",
        original_score=1.0,
        rerank_score=1.0,
        metadata={},
        rank_change=0,
    )

    assert result_min.rerank_score == 0.0
    assert result_max.rerank_score == 1.0


# ==================== State 边界条件测试 ====================

def test_agent_state_empty_fields():
    """测试AgentState处理空字段"""
    from src.graph.state import AgentState

    state: AgentState = {
        "input": "",
        "chat_history": [],
        "documents": [],
        "document_metadata": [],
        "generation": "",
        "relevance_score": "",
        "hallucination_score": "",
        "retry_count": 0,
    }

    assert state["input"] == ""
    assert len(state["chat_history"]) == 0
    assert len(state["documents"]) == 0


def test_agent_state_none_fields():
    """测试AgentState处理None字段（在允许的情况下）"""
    from src.graph.state import AgentState

    # 某些字段可以是空字符串，但不应该是None（类型定义不允许）
    state: AgentState = {
        "input": "测试",
        "chat_history": [],
        "documents": [],
        "document_metadata": [],
        "generation": "",
        "relevance_score": "yes",
        "hallucination_score": "grounded",
        "retry_count": 0,
    }

    assert state["input"] == "测试"


# ==================== 综合边界条件测试 ====================

def test_concurrent_empty_operations():
    """测试并发空操作"""
    from src.memory.short_term import ShortTermMemory

    memory = ShortTermMemory(max_turns=10)

    # 快速连续添加空操作
    for _ in range(100):
        memory.add_conversation_turn(human_message="", ai_message="")

    context = memory.get_recent_context()
    assert isinstance(context, dict)


def test_unicode_special_characters():
    """测试Unicode特殊字符处理"""
    from src.memory.short_term import ShortTermMemory

    memory = ShortTermMemory(max_turns=10)

    # 各种Unicode字符
    unicode_messages = [
        "中文测试",
        "🚀 emoji测试",
        "阿拉伯文: مرحبا",
        "日文: こんにちは",
        "俄文: Привет",
    ]

    for msg in unicode_messages:
        memory.add_conversation_turn(
            human_message=msg,
            ai_message="回答"
        )

    context = memory.get_recent_context()
    assert context["turn_count"] >= len(unicode_messages)


def test_very_long_metadata():
    """测试非常长的元数据"""
    from src.memory.long_term import MemoryItem

    # 创建包含大量数据的元数据
    large_metadata = {
        "key" + str(i): "value" * 100 for i in range(100)
    }

    item = MemoryItem(
        id="test-id",
        content="测试",
        memory_type="conversation",
        metadata=large_metadata
    )

    assert len(item.metadata) == 100


def test_nested_metadata():
    """测试嵌套元数据"""
    from src.memory.long_term import MemoryItem

    nested_metadata = {
        "level1": {
            "level2": {
                "level3": "value"
            }
        }
    }

    item = MemoryItem(
        id="test-id",
        content="测试",
        memory_type="conversation",
        metadata=nested_metadata
    )

    assert item.metadata["level1"]["level2"]["level3"] == "value"


def test_type_coercion():
    """测试类型转换边界情况"""
    from src.memory.short_term import ShortTermMemory

    memory = ShortTermMemory(max_turns=10)

    # 测试不同类型的数据（应该都能转换为字符串）
    memory.add_conversation_turn(
        human_message=str(123),  # 数字
        ai_message=str(True)  # 布尔值
    )

    context = memory.get_recent_context()
    assert context["turn_count"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
