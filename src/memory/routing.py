"""
自适应路由逻辑模块

实现智能记忆路由决策：
- 决策何时使用短期记忆
- 决策何时从长期记忆检索
- 决策是否需要混合检索
- 基于相关性评分、上下文状态等动态路由
"""

import time
from typing import List, Dict, Any, Optional, Tuple, Literal
from dataclasses import dataclass, field
from enum import Enum

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from ..config import get_settings
from .short_term import ShortTermMemory
from .long_term import LongTermMemory, MemoryItem


class RouteDecision(str, Enum):
    """路由决策结果"""
    SHORT_TERM_ONLY = "short_term_only"  # 仅使用短期记忆
    LONG_TERM_ONLY = "long_term_only"  # 仅从长期记忆检索
    HYBRID = "hybrid"  # 混合使用短期和长期记忆
    ARCHIVE_AND_RETRIEVE = "archive_and_retrieve"  # 归档短期记忆后检索


@dataclass
class RoutingContext:
    """路由上下文信息"""
    query: str
    short_term_memory: ShortTermMemory
    long_term_memory: LongTermMemory
    session_id: Optional[str] = None
    recent_relevance_score: Optional[float] = None  # 最近一次的相关性评分
    previous_retrieval_time: Optional[float] = None  # 上次检索时间
    query_complexity: Optional[float] = None  # 查询复杂度评分


@dataclass
class RoutingResult:
    """路由结果"""
    decision: RouteDecision
    reasoning: str
    confidence: float  # 决策置信度 (0-1)
    short_term_context: Optional[Dict[str, Any]] = None
    long_term_results: Optional[List[MemoryItem]] = None
    should_archive: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryRouter:
    """
    记忆路由器
    
    智能决策何时使用短期记忆，何时从长期记忆检索，
    以及是否需要混合使用两种记忆。
    """

    def __init__(
        self,
        short_term_threshold: Optional[int] = None,
        long_term_trigger: Optional[float] = None,
        relevance_threshold: Optional[float] = None,
        query_complexity_threshold: float = 0.6,
    ):
        """
        初始化记忆路由器
        
        Args:
            short_term_threshold: 短期记忆轮数阈值
            long_term_trigger: 长期记忆触发阈值（相关性评分）
            relevance_threshold: 相关性阈值（低于此值需要长期检索）
            query_complexity_threshold: 查询复杂度阈值
        """
        settings = get_settings()
        self.short_term_threshold = short_term_threshold or settings.short_term_threshold
        self.long_term_trigger = long_term_trigger or settings.long_term_trigger
        self.relevance_threshold = relevance_threshold or 0.7
        self.query_complexity_threshold = query_complexity_threshold

        logger.info(
            f"记忆路由器初始化: "
            f"short_term_threshold={self.short_term_threshold}, "
            f"long_term_trigger={self.long_term_trigger}, "
            f"relevance_threshold={self.relevance_threshold}"
        )

    def route(
        self,
        context: RoutingContext,
    ) -> RoutingResult:
        """
        执行路由决策
        
        Args:
            context: 路由上下文
        
        Returns:
            路由决策结果
        """
        query = context.query
        short_memory = context.short_term_memory
        long_memory = context.long_term_memory

        # 获取短期记忆状态
        short_term_stats = short_memory.get_stats()
        turn_count = short_term_stats.get("turn_count", 0)

        # 评估查询复杂度
        query_complexity = self._assess_query_complexity(query)

        # 评估是否需要历史上下文
        needs_history = self._needs_historical_context(query)

        # 决策流程
        decision, reasoning, confidence = self._make_decision(
            context=context,
            turn_count=turn_count,
            query_complexity=query_complexity,
            needs_history=needs_history,
        )

        # 执行路由并获取结果
        result = self._execute_route(
            decision=decision,
            context=context,
            reasoning=reasoning,
            confidence=confidence,
        )

        logger.info(
            f"路由决策: {decision.value} | "
            f"置信度: {confidence:.2f} | "
            f"轮数: {turn_count} | "
            f"推理: {reasoning}"
        )

        return result

    def _make_decision(
        self,
        context: RoutingContext,
        turn_count: int,
        query_complexity: float,
        needs_history: bool,
    ) -> Tuple[RouteDecision, str, float]:
        """
        做出路由决策
        
        Returns:
            (决策, 推理说明, 置信度)
        """
        query = context.query
        recent_score = context.recent_relevance_score
        
        # 确保 recent_score 是浮点数类型
        if recent_score is not None:
            try:
                recent_score = float(recent_score)
            except (ValueError, TypeError):
                # 如果无法转换，设为 None
                recent_score = None

        # 规则 1: 短期记忆为空，必须使用长期记忆
        if turn_count == 0:
            return (
                RouteDecision.LONG_TERM_ONLY,
                "短期记忆为空，需要从长期记忆检索",
                0.95,
            )

        # 规则 2: 短期记忆达到阈值，需要归档并检索长期记忆
        if turn_count >= self.short_term_threshold:
            return (
                RouteDecision.ARCHIVE_AND_RETRIEVE,
                f"短期记忆达到阈值 ({turn_count} >= {self.short_term_threshold})，需要归档并检索长期记忆",
                0.90,
            )

        # 规则 3: 最近相关性评分低于阈值，需要从长期记忆检索
        if recent_score is not None and recent_score < self.long_term_trigger:
            return (
                RouteDecision.HYBRID,
                f"最近相关性评分较低 ({recent_score:.2f} < {self.long_term_trigger})，需要补充长期记忆",
                0.85,
            )

        # 规则 4: 查询复杂度高，可能需要长期记忆
        if query_complexity > self.query_complexity_threshold:
            return (
                RouteDecision.HYBRID,
                f"查询复杂度较高 ({query_complexity:.2f} > {self.query_complexity_threshold})，建议混合检索",
                0.75,
            )

        # 规则 5: 需要历史上下文
        if needs_history:
            return (
                RouteDecision.HYBRID,
                "查询需要历史上下文，建议混合检索",
                0.80,
            )

        # 规则 6: 简单查询，优先使用短期记忆
        if query_complexity < 0.3 and turn_count > 0:
            return (
                RouteDecision.SHORT_TERM_ONLY,
                "简单查询且短期记忆可用，仅使用短期记忆",
                0.85,
            )

        # 默认：使用混合检索
        return (
            RouteDecision.HYBRID,
            "默认策略：混合使用短期和长期记忆以确保完整性",
            0.70,
        )

    def _execute_route(
        self,
        decision: RouteDecision,
        context: RoutingContext,
        reasoning: str,
        confidence: float,
    ) -> RoutingResult:
        """执行路由决策"""
        query = context.query
        short_memory = context.short_term_memory
        long_memory = context.long_term_memory
        session_id = context.session_id

        short_term_context = None
        long_term_results = None
        should_archive = False

        if decision == RouteDecision.SHORT_TERM_ONLY:
            # 仅使用短期记忆
            short_term_context = short_memory.get_recent_context(
                include_entities=True
            )

        elif decision == RouteDecision.LONG_TERM_ONLY:
            # 仅从长期记忆检索
            long_term_results = long_memory.search(
                query=query,
                top_k=5,
                session_id=session_id,
            )

        elif decision == RouteDecision.HYBRID:
            # 混合检索
            short_term_context = short_memory.get_recent_context(
                include_entities=True
            )
            long_term_results = long_memory.search(
                query=query,
                top_k=5,
                session_id=session_id,
            )

        elif decision == RouteDecision.ARCHIVE_AND_RETRIEVE:
            # 归档并检索
            should_archive = True

            # 归档短期记忆
            short_term_context_data = short_memory.get_recent_context(
                include_entities=False
            )
            archived_ids = long_memory.archive_from_short_term(
                conversation_turns=short_term_context_data.get("conversation", []),
                session_id=session_id,
                deduplicate=True,
                semantic_dedup=True,  # 启用语义相似度去重
            )

            # 检索长期记忆
            long_term_results = long_memory.search(
                query=query,
                top_k=5,
                session_id=session_id,
            )

            # 清理短期记忆（可选：只保留最近几轮）
            short_memory.compress(method="summary")

            short_term_context = short_memory.get_recent_context(
                include_entities=True
            )

        return RoutingResult(
            decision=decision,
            reasoning=reasoning,
            confidence=confidence,
            short_term_context=short_term_context,
            long_term_results=long_term_results,
            should_archive=should_archive,
            metadata={
                "query": query,
                "timestamp": time.time(),
                "session_id": session_id,
            },
        )

    def _assess_query_complexity(self, query: str) -> float:
        """
        评估查询复杂度
        
        Args:
            query: 查询文本
        
        Returns:
            复杂度评分 (0-1)，越高越复杂
        """
        # 简单的启发式规则
        complexity = 0.0

        # 长度因素
        length_score = min(len(query) / 200.0, 1.0)
        complexity += length_score * 0.2

        # 关键词因素（疑问词、复杂词等）
        complex_keywords = [
            "为什么", "如何", "怎样", "解释", "分析", "比较",
            "why", "how", "explain", "analyze", "compare",
            "差异", "区别", "关系", "影响",
        ]
        keyword_count = sum(1 for kw in complex_keywords if kw in query.lower())
        keyword_score = min(keyword_count / 3.0, 1.0)
        complexity += keyword_score * 0.3

        # 历史引用（提到"之前"、"刚才"等）
        history_keywords = ["之前", "刚才", "之前提到", "以前", "之前说过", "before", "previously"]
        has_history_ref = any(kw in query.lower() for kw in history_keywords)
        if has_history_ref:
            complexity += 0.3

        # 实体数量（简单估算：大写单词、引号内容）
        import re
        entities = len(re.findall(r'["\']([^"\']+)["\']|[A-Z][a-z]+', query))
        entity_score = min(entities / 5.0, 1.0)
        complexity += entity_score * 0.2

        return min(complexity, 1.0)

    def _needs_historical_context(self, query: str) -> bool:
        """
        判断查询是否需要历史上下文
        
        Args:
            query: 查询文本
        
        Returns:
            是否需要历史上下文
        """
        history_indicators = [
            "之前", "刚才", "之前提到", "以前", "之前说过",
            "之前的问题", "前面的对话", "刚才说的",
            "before", "previously", "earlier", "mentioned",
            "remember", "recall", "之前记得",
        ]

        query_lower = query.lower()
        return any(indicator in query_lower for indicator in history_indicators)

    def should_retry_with_long_term(
        self,
        relevance_score: float,
        retry_count: int,
    ) -> bool:
        """
        判断是否应该使用长期记忆重试
        
        Args:
            relevance_score: 当前相关性评分
            retry_count: 重试次数
        
        Returns:
            是否应该重试
        """
        # 如果相关性很低，且未重试过，建议重试
        if relevance_score < self.long_term_trigger and retry_count == 0:
            return True

        # 如果相关性仍然很低，且重试次数少于2，继续重试
        if relevance_score < self.relevance_threshold and retry_count < 2:
            return True

        return False

    def get_routing_summary(self, result: RoutingResult) -> Dict[str, Any]:
        """
        获取路由结果摘要
        
        Args:
            result: 路由结果
        
        Returns:
            摘要信息
        """
        summary = {
            "decision": result.decision.value,
            "reasoning": result.reasoning,
            "confidence": result.confidence,
            "has_short_term": result.short_term_context is not None,
            "has_long_term": result.long_term_results is not None,
            "should_archive": result.should_archive,
        }

        if result.short_term_context:
            summary["short_term_turns"] = result.short_term_context.get(
                "turn_count", 0
            )

        if result.long_term_results:
            summary["long_term_count"] = len(result.long_term_results)
            if result.long_term_results:
                summary["max_relevance"] = max(
                    (r.relevance_score or 0.0 for r in result.long_term_results),
                    default=0.0,
                )

        return summary
