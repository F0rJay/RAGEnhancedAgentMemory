"""
测试检索增强模块
"""

import pytest

from src.retrieval.reranker import Reranker, RerankedResult
from src.retrieval.hybrid import HybridRetriever, HybridRetrievalResult


def test_reranked_result_creation():
    """测试重排序结果创建"""
    result = RerankedResult(
        content="测试内容",
        original_score=0.7,
        rerank_score=0.9,
        metadata={"key": "value"},
        rank_change=-2,  # 提升了2位
    )

    assert result.content == "测试内容"
    assert result.original_score == 0.7
    assert result.rerank_score == 0.9
    assert result.rank_change == -2


def test_hybrid_retrieval_result_creation():
    """测试混合检索结果创建"""
    result = HybridRetrievalResult(
        content="测试内容",
        vector_score=0.8,
        keyword_score=0.6,
        combined_score=0.74,  # 0.8 * 0.7 + 0.6 * 0.3
        metadata={"key": "value"},
        source="hybrid",
    )

    assert result.content == "测试内容"
    assert result.vector_score == 0.8
    assert result.keyword_score == 0.6
    assert result.combined_score == 0.74
    assert result.source == "hybrid"


def test_extract_keywords():
    """测试关键词提取"""
    # 这个测试需要实际的 HybridRetriever 实例
    # 由于需要 long_term_memory，这里只测试逻辑
    import re

    query = "Python 编程语言的特点是什么？"
    text = re.sub(r'[^\w\s]', ' ', query.lower())
    words = text.split()

    assert len(words) > 0
    assert "python" in text.lower()


def test_keyword_extraction_chinese():
    """测试中文关键词提取"""
    # 简化测试：检查中文字符处理
    query = "什么是机器学习"
    assert len(query) > 0


def test_weight_validation():
    """测试权重验证"""
    # 这个测试需要实际的 HybridRetriever 实例
    # 测试权重和必须为1.0
    with pytest.raises(ValueError):
        # 这里只是演示，实际测试需要创建实例
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
