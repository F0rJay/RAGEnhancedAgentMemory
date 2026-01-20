"""
测试自适应路由逻辑模块
"""

import pytest

from src.memory.routing import (
    MemoryRouter,
    RouteDecision,
    RoutingContext,
    RoutingResult,
)


def test_route_decision_enum():
    """测试路由决策枚举"""
    assert RouteDecision.SHORT_TERM_ONLY == "short_term_only"
    assert RouteDecision.LONG_TERM_ONLY == "long_term_only"
    assert RouteDecision.HYBRID == "hybrid"
    assert RouteDecision.ARCHIVE_AND_RETRIEVE == "archive_and_retrieve"


def test_memory_router_initialization():
    """测试记忆路由器初始化"""
    router = MemoryRouter(
        short_term_threshold=10,
        long_term_trigger=0.7,
    )

    assert router.short_term_threshold == 10
    assert router.long_term_trigger == 0.7
    assert router.relevance_threshold == 0.7


def test_assess_query_complexity():
    """测试查询复杂度评估"""
    router = MemoryRouter()

    # 简单查询
    simple_query = "你好"
    complexity_simple = router._assess_query_complexity(simple_query)
    assert 0.0 <= complexity_simple <= 1.0

    # 复杂查询
    complex_query = "为什么之前的对话中提到的问题没有得到解决？请分析一下原因并解释差异。"
    complexity_complex = router._assess_query_complexity(complex_query)
    assert complexity_complex > complexity_simple


def test_needs_historical_context():
    """测试历史上下文需求判断"""
    router = MemoryRouter()

    # 需要历史上下文的查询
    history_query = "之前提到的那个问题现在怎么样了？"
    assert router._needs_historical_context(history_query) is True

    # 不需要历史上下文的查询
    normal_query = "今天天气怎么样？"
    assert router._needs_historical_context(normal_query) is False


def test_should_retry_with_long_term():
    """测试长期记忆重试判断"""
    router = MemoryRouter()

    # 低相关性，应该重试
    assert router.should_retry_with_long_term(0.5, 0) is True

    # 高相关性，不需要重试
    assert router.should_retry_with_long_term(0.9, 0) is False

    # 低相关性但已重试多次，不再重试
    assert router.should_retry_with_long_term(0.5, 3) is False


def test_routing_result_creation():
    """测试路由结果创建"""
    result = RoutingResult(
        decision=RouteDecision.HYBRID,
        reasoning="测试推理",
        confidence=0.8,
    )

    assert result.decision == RouteDecision.HYBRID
    assert result.reasoning == "测试推理"
    assert result.confidence == 0.8
    assert result.should_archive is False


def test_get_routing_summary():
    """测试路由摘要生成"""
    router = MemoryRouter()

    result = RoutingResult(
        decision=RouteDecision.HYBRID,
        reasoning="测试",
        confidence=0.8,
    )

    summary = router.get_routing_summary(result)
    assert "decision" in summary
    assert "reasoning" in summary
    assert "confidence" in summary
    assert summary["decision"] == RouteDecision.HYBRID.value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
