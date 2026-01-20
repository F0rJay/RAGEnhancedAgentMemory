"""
混合检索模块

实现混合检索策略，结合向量检索和关键词检索，提升检索效果。
"""

import re
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import Counter

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from ..config import get_settings
from .reranker import Reranker, RerankedResult
from ..memory.long_term import LongTermMemory, MemoryItem


@dataclass
class HybridRetrievalResult:
    """混合检索结果"""
    content: str
    vector_score: float  # 向量检索评分
    keyword_score: float  # 关键词检索评分
    combined_score: float  # 综合评分
    metadata: Dict[str, Any]
    source: str  # 来源：vector/keyword/hybrid


class HybridRetriever:
    """
    混合检索器
    
    结合向量检索（语义相似度）和关键词检索（精确匹配），
    提供更全面和准确的检索结果。
    """

    def __init__(
        self,
        long_term_memory: LongTermMemory,
        reranker: Optional[Reranker] = None,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        use_rerank: bool = True,
    ):
        """
        初始化混合检索器
        
        Args:
            long_term_memory: 长期记忆管理器（用于向量检索）
            reranker: 重排序器（可选）
            vector_weight: 向量检索权重
            keyword_weight: 关键词检索权重
            use_rerank: 是否使用重排序
        """
        if abs(vector_weight + keyword_weight - 1.0) > 0.01:
            raise ValueError("vector_weight 和 keyword_weight 之和必须等于 1.0")

        self.long_term_memory = long_term_memory
        self.reranker = reranker
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.use_rerank = use_rerank and reranker is not None

        logger.info(
            f"混合检索器初始化: "
            f"vector_weight={vector_weight}, "
            f"keyword_weight={keyword_weight}, "
            f"use_rerank={use_rerank}"
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        rerank_top_k: Optional[int] = None,
        session_id: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        keyword_expansion: bool = True,
    ) -> List[HybridRetrievalResult]:
        """
        执行混合检索
        
        Args:
            query: 查询文本
            top_k: 初始检索 Top-K
            rerank_top_k: 重排序后的 Top-K
            session_id: 会话 ID
            metadata_filter: 元数据过滤条件
            keyword_expansion: 是否进行关键词扩展
        
        Returns:
            混合检索结果列表
        """
        start_time = time.time()

        # 1. 向量检索
        vector_results = self._vector_search(
            query=query,
            top_k=top_k,
            session_id=session_id,
            metadata_filter=metadata_filter,
        )

        # 2. 关键词检索
        keywords = self._extract_keywords(query, expand=keyword_expansion)
        keyword_results = self._keyword_search(
            keywords=keywords,
            top_k=top_k,
            session_id=session_id,
            metadata_filter=metadata_filter,
        )

        # 3. 合并结果
        combined_results = self._combine_results(
            vector_results=vector_results,
            keyword_results=keyword_results,
            query=query,
        )

        # 4. 重排序（可选）
        if self.use_rerank and self.reranker:
            reranked_results = self.reranker.rerank(
                query=query,
                documents=[r.content for r in combined_results],
                scores=[r.combined_score for r in combined_results],
                metadatas=[r.metadata for r in combined_results],
                top_k=rerank_top_k or top_k,
            )

            # 转换为 HybridRetrievalResult
            final_results = []
            for reranked in reranked_results:
                # 找到原始结果
                original = next(
                    (r for r in combined_results if r.content == reranked.content),
                    None,
                )
                if original:
                    final_results.append(
                        HybridRetrievalResult(
                            content=reranked.content,
                            vector_score=original.vector_score,
                            keyword_score=original.keyword_score,
                            combined_score=reranked.rerank_score,  # 使用重排序评分
                            metadata=reranked.metadata,
                            source=original.source,
                        )
                    )
            combined_results = final_results

        retrieval_time = time.time() - start_time

        logger.info(
            f"混合检索完成: "
            f"向量结果={len(vector_results)}, "
            f"关键词结果={len(keyword_results)}, "
            f"合并结果={len(combined_results)}, "
            f"耗时={retrieval_time:.3f}s"
        )

        return combined_results

    def _vector_search(
        self,
        query: str,
        top_k: int,
        session_id: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        向量检索
        
        Returns:
            [(content, score, metadata), ...]
        """
        results = self.long_term_memory.search(
            query=query,
            top_k=top_k,
            session_id=session_id,
            metadata_filter=metadata_filter,
        )

        return [
            (item.content, item.relevance_score or 0.0, item.metadata)
            for item in results
        ]

    def _extract_keywords(
        self,
        query: str,
        expand: bool = True,
    ) -> List[str]:
        """
        从查询中提取关键词
        
        Args:
            query: 查询文本
            expand: 是否进行关键词扩展
        
        Returns:
            关键词列表
        """
        # 移除标点符号
        text = re.sub(r'[^\w\s]', ' ', query.lower())

        # 简单分词（中文和英文）
        # 中文：按字符（简化处理）
        # 英文：按单词
        words = []
        for segment in text.split():
            # 英文单词
            if segment.isascii():
                words.append(segment)
            else:
                # 中文字符（每个字符作为一个词）
                words.extend(list(segment))

        # 移除停用词
        stopwords = {
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "is", "are", "was", "were", "been", "be",
        }
        keywords = [w for w in words if w not in stopwords and len(w) > 1]

        # 去重
        keywords = list(dict.fromkeys(keywords))

        # 关键词扩展（可选）
        if expand and keywords:
            expanded = []
            for kw in keywords:
                expanded.append(kw)
                # 简单的同义词扩展（实际应用中可以使用词向量或词典）
                # 这里仅作示例
            keywords = expanded

        return keywords[:10]  # 限制关键词数量

    def _keyword_search(
        self,
        keywords: List[str],
        top_k: int,
        session_id: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        关键词检索
        
        Returns:
            [(content, score, metadata), ...]
        """
        if not keywords:
            return []

        # 执行多个关键词查询
        all_results: Dict[str, Tuple[float, Dict[str, Any]]] = {}

        for keyword in keywords:
            # 使用向量数据库的关键词检索（简化版：使用关键词作为查询）
            results = self.long_term_memory.search(
                query=keyword,
                top_k=top_k * 2,  # 检索更多结果以合并
                session_id=session_id,
                metadata_filter=metadata_filter,
            )

            # 计算关键词匹配分数
            for item in results:
                content_lower = item.content.lower()
                keyword_lower = keyword.lower()

                # 简单的关键词匹配评分
                keyword_count = content_lower.count(keyword_lower)
                match_score = min(keyword_count / 3.0, 1.0)  # 归一化

                # 结合向量检索评分和关键词匹配评分
                combined_score = (item.relevance_score or 0.0) * 0.5 + match_score * 0.5

                # 合并结果
                if item.content in all_results:
                    # 取最高分
                    existing_score, metadata = all_results[item.content]
                    if combined_score > existing_score:
                        all_results[item.content] = (combined_score, item.metadata)
                else:
                    all_results[item.content] = (combined_score, item.metadata)

        # 排序并返回 Top-K
        sorted_results = sorted(
            all_results.items(),
            key=lambda x: x[1][0],
            reverse=True,
        )[:top_k]

        return [(content, score, metadata) for content, (score, metadata) in sorted_results]

    def _combine_results(
        self,
        vector_results: List[Tuple[str, float, Dict[str, Any]]],
        keyword_results: List[Tuple[str, float, Dict[str, Any]]],
        query: str,
    ) -> List[HybridRetrievalResult]:
        """
        合并向量检索和关键词检索结果
        
        Returns:
            合并后的结果列表
        """
        # 创建结果字典
        result_dict: Dict[str, HybridRetrievalResult] = {}

        # 添加向量检索结果
        for content, score, metadata in vector_results:
            result_dict[content] = HybridRetrievalResult(
                content=content,
                vector_score=score,
                keyword_score=0.0,
                combined_score=score * self.vector_weight,
                metadata=metadata,
                source="vector",
            )

        # 合并关键词检索结果
        for content, score, metadata in keyword_results:
            if content in result_dict:
                # 更新现有结果
                existing = result_dict[content]
                existing.keyword_score = score
                existing.combined_score = (
                    existing.vector_score * self.vector_weight
                    + score * self.keyword_weight
                )
                existing.source = "hybrid"
            else:
                # 添加新结果
                result_dict[content] = HybridRetrievalResult(
                    content=content,
                    vector_score=0.0,
                    keyword_score=score,
                    combined_score=score * self.keyword_weight,
                    metadata=metadata,
                    source="keyword",
                )

        # 按综合评分排序
        sorted_results = sorted(
            result_dict.values(),
            key=lambda x: x.combined_score,
            reverse=True,
        )

        return sorted_results

    def retrieve_with_fallback(
        self,
        query: str,
        top_k: int = 10,
        session_id: Optional[str] = None,
    ) -> List[HybridRetrievalResult]:
        """
        带降级策略的检索（向量检索失败时使用关键词检索）
        
        Args:
            query: 查询文本
            top_k: Top-K
            session_id: 会话 ID
        
        Returns:
            检索结果列表
        """
        try:
            # 尝试混合检索
            results = self.retrieve(
                query=query,
                top_k=top_k,
                session_id=session_id,
            )
            return results
        except Exception as e:
            logger.warning(f"混合检索失败，降级到关键词检索: {e}")
            # 降级到关键词检索
            keywords = self._extract_keywords(query, expand=False)
            keyword_results = self._keyword_search(
                keywords=keywords,
                top_k=top_k,
                session_id=session_id,
            )

            return [
                HybridRetrievalResult(
                    content=content,
                    vector_score=0.0,
                    keyword_score=score,
                    combined_score=score,
                    metadata=metadata,
                    source="keyword",
                )
                for content, score, metadata in keyword_results
            ]
