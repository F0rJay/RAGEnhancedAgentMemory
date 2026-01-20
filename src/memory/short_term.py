"""
短期记忆管理模块

实现短期记忆（Short-Term/Working Memory）：
- 维护当前会话的上下文连贯性
- 最近 N 轮对话的滑动窗口
- 关键实体缓存
- 动态压缩机制防止上下文窗口爆炸
"""

import time
from typing import List, Dict, Any, Optional, Set, TYPE_CHECKING
from collections import deque
from dataclasses import dataclass, field

# 类型检查时导入，避免循环依赖
if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

try:
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
except ImportError as e:
    raise ImportError(
        "langchain-core 未安装。请运行: pip install langchain-core>=0.3.0"
    ) from e

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from ..config import get_settings


@dataclass
class ConversationTurn:
    """单轮对话记录"""
    human_message: str
    ai_message: str
    timestamp: float = field(default_factory=time.time)
    entities: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KeyEntity:
    """关键实体"""
    name: str
    context: str  # 实体出现的上下文
    importance_score: float  # 重要性评分 (0-1)
    first_seen: float  # 首次出现时间
    last_seen: float  # 最后出现时间
    mention_count: int = 1  # 提及次数


class ShortTermMemory:
    """
    短期记忆管理器
    
    负责维护当前会话的上下文连贯性，使用滑动窗口机制管理最近 N 轮对话，
    并提取和缓存关键实体信息。
    """

    def __init__(
        self,
        max_turns: Optional[int] = None,
        max_tokens: Optional[int] = None,
        compression_threshold: int = 15,
        entity_importance_threshold: float = 0.5,
    ):
        """
        初始化短期记忆管理器
        
        Args:
            max_turns: 最大保留轮数（None 表示使用配置值）
            max_tokens: 最大 token 数（用于动态压缩，None 表示不限制）
            compression_threshold: 触发压缩的轮数阈值
            entity_importance_threshold: 实体重要性阈值
        """
        settings = get_settings()
        self.max_turns = max_turns or settings.short_term_threshold
        self.max_tokens = max_tokens
        self.compression_threshold = compression_threshold
        self.entity_importance_threshold = entity_importance_threshold

        # 滑动窗口缓冲区
        self.conversation_buffer: deque[ConversationTurn] = deque(maxlen=self.max_turns)

        # 关键实体缓存（实体名 -> KeyEntity）
        self.key_entities: Dict[str, KeyEntity] = {}

        # 会话元数据
        self.session_id: Optional[str] = None
        self.created_at: float = time.time()
        self.last_accessed: float = time.time()

        logger.info(
            f"短期记忆管理器初始化: max_turns={self.max_turns}, "
            f"compression_threshold={self.compression_threshold}"
        )

    def add_conversation_turn(
        self,
        human_message: str,
        ai_message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        添加一轮对话到短期记忆
        
        Args:
            human_message: 用户消息
            ai_message: AI 回答
            metadata: 额外的元数据
        """
        # 提取关键实体
        entities = self._extract_entities(human_message, ai_message)

        # 创建对话轮次
        turn = ConversationTurn(
            human_message=human_message,
            ai_message=ai_message,
            timestamp=time.time(),
            entities=entities,
            metadata=metadata or {},
        )

        # 添加到缓冲区
        self.conversation_buffer.append(turn)

        # 更新实体缓存
        self._update_entity_cache(entities, human_message, ai_message)

        # 更新最后访问时间
        self.last_accessed = time.time()

        # 检查是否需要压缩
        if len(self.conversation_buffer) >= self.compression_threshold:
            logger.warning(
                f"对话轮数达到压缩阈值 ({self.compression_threshold})，执行压缩"
            )
            self.compress()

        logger.debug(
            f"添加对话轮次: 当前轮数={len(self.conversation_buffer)}, "
            f"实体数={len(self.key_entities)}"
        )

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """
        从 LangChain 消息列表添加对话
        
        Args:
            messages: LangChain 消息列表
        """
        human_msg = None
        ai_msg = None

        for msg in messages:
            if isinstance(msg, HumanMessage):
                human_msg = msg.content
            elif isinstance(msg, AIMessage):
                ai_msg = msg.content

            # 当收集到一对消息时，添加到记忆
            if human_msg and ai_msg:
                self.add_conversation_turn(human_msg, ai_msg)
                human_msg = None
                ai_msg = None

    def get_recent_context(
        self, 
        n_turns: Optional[int] = None,
        include_entities: bool = True,
    ) -> Dict[str, Any]:
        """
        获取最近的上下文
        
        Args:
            n_turns: 返回最近 N 轮（None 表示全部）
            include_entities: 是否包含关键实体信息
        
        Returns:
            包含最近对话和关键实体的字典
        """
        n_turns = n_turns or len(self.conversation_buffer)

        # 获取最近的对话轮次
        recent_turns = list(self.conversation_buffer)[-n_turns:]

        # 构建上下文
        context = {
            "conversation": [
                {
                    "human": turn.human_message,
                    "ai": turn.ai_message,
                    "timestamp": turn.timestamp,
                }
                for turn in recent_turns
            ],
            "turn_count": len(recent_turns),
            "total_turns": len(self.conversation_buffer),
        }

        # 添加关键实体
        if include_entities:
            context["key_entities"] = self._get_important_entities()

        self.last_accessed = time.time()
        return context

    def get_context_for_llm(self, max_tokens: Optional[int] = None) -> str:
        """
        获取格式化的上下文字符串，用于 LLM 输入
        
        Args:
            max_tokens: 最大 token 数（粗略估计）
        
        Returns:
            格式化的上下文字符串
        """
        context_parts = []

        # 添加关键实体摘要
        important_entities = self._get_important_entities()
        if important_entities:
            entity_summary = "关键实体：\n"
            for entity in important_entities[:10]:  # 最多显示10个实体
                entity_summary += f"- {entity['name']}: {entity['context']}\n"
            context_parts.append(entity_summary)

        # 添加最近对话历史
        recent_context = self.get_recent_context(include_entities=False)
        conversation_text = "\n\n".join(
            f"用户: {turn['human']}\n助手: {turn['ai']}"
            for turn in recent_context["conversation"]
        )
        context_parts.append(f"对话历史：\n{conversation_text}")

        result = "\n\n".join(context_parts)

        # 简单的 token 估算（粗略：1 token ≈ 4 字符）
        if max_tokens and len(result) > max_tokens * 4:
            # 如果超过限制，进行截断
            result = self._truncate_to_tokens(result, max_tokens)

        return result

    def compress(self, method: str = "summary") -> None:
        """
        压缩短期记忆
        
        当对话轮数过多时，通过压缩减少上下文长度。
        
        Args:
            method: 压缩方法 ("summary", "entity_based", "time_based")
        """
        if len(self.conversation_buffer) < self.compression_threshold:
            return

        logger.info(f"执行短期记忆压缩: method={method}, 当前轮数={len(self.conversation_buffer)}")

        if method == "summary":
            self._compress_by_summary()
        elif method == "entity_based":
            self._compress_by_entities()
        elif method == "time_based":
            self._compress_by_time()
        else:
            logger.warning(f"未知的压缩方法: {method}")

    def _compress_by_summary(self) -> None:
        """通过摘要压缩：保留最近的对话，将较老的对话合并为摘要"""
        if len(self.conversation_buffer) <= self.max_turns:
            return

        # 保留最近的一半对话
        keep_count = self.max_turns // 2
        recent_turns = list(self.conversation_buffer)[-keep_count:]
        old_turns = list(self.conversation_buffer)[:-keep_count]

        # 为较老的对话生成摘要（简化版：拼接关键信息）
        if old_turns:
            summary = self._generate_summary(old_turns)
            # 创建一个摘要对话轮次
            summary_turn = ConversationTurn(
                human_message="[历史对话摘要]",
                ai_message=summary,
                timestamp=old_turns[0].timestamp,
            )
            # 清空缓冲区并重建
            self.conversation_buffer.clear()
            self.conversation_buffer.append(summary_turn)
            self.conversation_buffer.extend(recent_turns)

        logger.info(f"摘要压缩完成: 保留 {len(self.conversation_buffer)} 轮对话")

    def _compress_by_entities(self) -> None:
        """基于实体压缩：保留包含重要实体的对话"""
        important_entity_names = {
            entity.name
            for entity in self.key_entities.values()
            if entity.importance_score >= self.entity_importance_threshold
        }

        # 过滤：保留包含重要实体的对话或最近的对话
        keep_count = self.max_turns // 2
        filtered_turns = []

        # 首先保留最近的对话
        for turn in list(self.conversation_buffer)[-keep_count:]:
            filtered_turns.append(turn)

        # 然后保留包含重要实体的对话
        for turn in list(self.conversation_buffer)[:-keep_count]:
            if turn.entities & important_entity_names:
                filtered_turns.append(turn)

        # 去重并排序
        seen = set()
        unique_turns = []
        for turn in filtered_turns:
            turn_id = (turn.human_message, turn.ai_message)
            if turn_id not in seen:
                seen.add(turn_id)
                unique_turns.append(turn)

        # 按时间排序
        unique_turns.sort(key=lambda t: t.timestamp)

        # 更新缓冲区
        self.conversation_buffer.clear()
        self.conversation_buffer.extend(unique_turns[-self.max_turns:])

        logger.info(f"实体压缩完成: 保留 {len(self.conversation_buffer)} 轮对话")

    def _compress_by_time(self) -> None:
        """基于时间压缩：保留最近时间范围内的对话"""
        current_time = time.time()
        time_window = 3600  # 保留最近1小时内的对话

        filtered_turns = [
            turn
            for turn in self.conversation_buffer
            if current_time - turn.timestamp < time_window
        ]

        self.conversation_buffer.clear()
        self.conversation_buffer.extend(filtered_turns)

        logger.info(f"时间压缩完成: 保留 {len(self.conversation_buffer)} 轮对话")

    def _extract_entities(self, human_msg: str, ai_msg: str) -> Set[str]:
        """
        从消息中提取关键实体
        
        这是一个简化版的实体提取，实际应用中可以使用 NER 模型。
        
        Args:
            human_msg: 用户消息
            ai_msg: AI 回答
        
        Returns:
            提取到的实体名称集合
        """
        entities = set()
        text = f"{human_msg} {ai_msg}"

        # 简单的启发式规则：提取大写开头的连续词、引号内容等
        # 实际应用中可以集成 spaCy、NER 模型等
        import re

        # 提取引号内容
        quoted = re.findall(r'"([^"]+)"', text)
        entities.update(quoted)

        # 提取大写开头的专有名词（简单模式）
        proper_nouns = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        entities.update(proper_nouns)

        # 提取数字相关的实体（如价格、日期等）
        # numbers = re.findall(r'\b\d+(?:\.\d+)?\s*(?:元|美元|年|月|日)\b', text)
        # entities.update(numbers)

        return entities

    def _update_entity_cache(
        self, 
        entities: Set[str], 
        human_msg: str, 
        ai_msg: str,
    ) -> None:
        """更新实体缓存"""
        current_time = time.time()
        context = f"{human_msg[:100]}... {ai_msg[:100]}..."  # 截断上下文

        for entity in entities:
            if entity in self.key_entities:
                # 更新现有实体
                entity_obj = self.key_entities[entity]
                entity_obj.last_seen = current_time
                entity_obj.mention_count += 1
                # 重要性评分：基于提及次数和时效性
                entity_obj.importance_score = min(
                    1.0,
                    entity_obj.mention_count / 5.0
                    + (1.0 - (current_time - entity_obj.first_seen) / 3600.0) * 0.3,
                )
            else:
                # 创建新实体
                self.key_entities[entity] = KeyEntity(
                    name=entity,
                    context=context,
                    importance_score=0.3,
                    first_seen=current_time,
                    last_seen=current_time,
                    mention_count=1,
                )

    def _get_important_entities(self, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        获取重要性最高的实体
        
        Args:
            top_k: 返回前 K 个实体
        
        Returns:
            实体信息列表
        """
        sorted_entities = sorted(
            self.key_entities.values(),
            key=lambda e: e.importance_score,
            reverse=True,
        )

        return [
            {
                "name": entity.name,
                "context": entity.context,
                "importance_score": entity.importance_score,
                "mention_count": entity.mention_count,
            }
            for entity in sorted_entities[:top_k]
        ]

    def _generate_summary(self, turns: List[ConversationTurn]) -> str:
        """
        为对话轮次生成摘要
        
        简化版：实际应用中可以使用 LLM 生成摘要。
        
        Args:
            turns: 需要摘要的对话轮次
        
        Returns:
            摘要文本
        """
        if not turns:
            return ""

        # 提取关键信息
        topics = []
        entities = set()

        for turn in turns:
            topics.append(turn.human_message[:50])  # 简化：取前50字符
            entities.update(turn.entities)

        summary = f"讨论了 {len(turns)} 轮对话，主要话题包括：{', '.join(topics[:5])}"
        if entities:
            summary += f"。涉及的关键实体：{', '.join(list(entities)[:5])}"

        return summary

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        截断文本到指定 token 数（粗略估计）
        
        Args:
            text: 原始文本
            max_tokens: 最大 token 数
        
        Returns:
            截断后的文本
        """
        # 粗略估算：1 token ≈ 4 字符
        max_chars = max_tokens * 4

        if len(text) <= max_chars:
            return text

        # 从后往前截断，保留最重要的部分
        truncated = "..." + text[-(max_chars - 3):]
        return truncated

    def clear(self) -> None:
        """清空短期记忆"""
        self.conversation_buffer.clear()
        self.key_entities.clear()
        logger.info("短期记忆已清空")

    def get_stats(self) -> Dict[str, Any]:
        """获取短期记忆统计信息"""
        return {
            "turn_count": len(self.conversation_buffer),
            "max_turns": self.max_turns,
            "entity_count": len(self.key_entities),
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "age_seconds": time.time() - self.created_at,
        }

    def set_session_id(self, session_id: str) -> None:
        """设置会话 ID"""
        self.session_id = session_id
