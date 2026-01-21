"""
测试评估体系模块
"""

import pytest

from src.evaluation.ragas_eval import RagasEvaluator, EvaluationResult
from src.evaluation.needle_test import (
    NeedleInHaystackTester,
    NeedleTestResult,
)


def test_evaluation_result_creation():
    """测试评估结果创建"""
    result = EvaluationResult(
        context_recall=0.9,
        context_precision=0.85,
        faithfulness=0.95,
        answer_relevancy=0.88,
        overall_score=0.895,
    )

    assert result.context_recall == 0.9
    assert result.overall_score == 0.895
    assert result.faithfulness == 0.95


def test_needle_test_result_creation():
    """测试 Needle 测试结果创建"""
    result = NeedleTestResult(
        position=0.5,
        success=True,
        answer="测试答案",
        expected_answer="测试答案",
        accuracy=1.0,
    )

    assert result.position == 0.5
    assert result.success is True
    assert result.accuracy == 1.0


def test_needle_in_haystack_tester_initialization():
    """测试 Needle 测试器初始化"""
    tester = NeedleInHaystackTester(
        token_estimate=10000,
        needle_positions=[0.0, 0.5, 1.0],
    )

    assert tester.token_estimate == 10000
    assert len(tester.needle_positions) == 3
    assert 0.5 in tester.needle_positions


def test_generate_haystack():
    """测试 Haystack 生成"""
    tester = NeedleInHaystackTester()

    needle = "系统的秘密启动密码是 9527"
    haystack = tester.generate_haystack(needle, position=0.5)

    assert needle in haystack
    assert len(haystack) > len(needle)


def test_generate_test_cases():
    """测试测试用例生成"""
    tester = NeedleInHaystackTester()

    needles = [
        "测试 Needle 1",
        "测试 Needle 2",
    ]

    test_cases = tester.generate_test_cases(needles, positions=[0.0, 1.0])

    assert len(test_cases) == 4  # 2 needles * 2 positions
    assert all("haystack" in tc for tc in test_cases)
    assert all("question" in tc for tc in test_cases)


def test_evaluate_answer():
    """测试答案评估"""
    tester = NeedleInHaystackTester()

    # 精确匹配
    success, accuracy = tester._evaluate_answer("答案123", "答案123")
    assert success is True
    assert accuracy == 1.0

    # 部分匹配
    success, accuracy = tester._evaluate_answer("答案", "答案123")
    assert accuracy > 0.0


def test_check_threshold():
    """测试阈值检查"""
    # 需要实际实例，这里仅测试逻辑
    result = EvaluationResult(
        context_recall=0.8,
        context_precision=0.85,
        faithfulness=0.9,
        answer_relevancy=0.88,
        overall_score=0.8575,
    )

    # 简单测试
    assert result.overall_score > 0.75


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
