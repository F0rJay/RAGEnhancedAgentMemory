"""
基线 Agent 实现

实现两种传统的记忆管理策略，作为性能对比的基线：
1. 滑动窗口 Agent：只保留最近 N 轮对话
2. 全量摘要 Agent：使用摘要压缩历史对话
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
from dataclasses import dataclass, field

try:
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
except ImportError:
    BaseMessage = None
    HumanMessage = None
    AIMessage = None

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """单轮对话记录"""
    human: str
    ai: str
    turn_number: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class BaselineMetrics:
    """基线系统性能指标"""
    total_turns: int = 0
    context_length: int = 0  # 当前上下文长度（token 估算）
    storage_bytes: int = 0  # 存储占用（字节）
    retrieval_latency: float = 0.0  # 检索延迟
    generation_latency: float = 0.0  # 生成延迟
    total_latency: float = 0.0  # 总延迟
    early_info_accessible: bool = False  # 能否访问早期信息


class SlidingWindowAgent:
    """
    滑动窗口 Agent（基线系统 1）
    
    只保留最近 N 轮对话，超出窗口的对话被丢弃。
    这是最常见的简单记忆管理策略。
    """
    
    def __init__(self, window_size: int = 10):
        """
        初始化滑动窗口 Agent
        
        Args:
            window_size: 滑动窗口大小（保留的对话轮数）
        """
        self.window_size = window_size
        self.conversation_history: deque[ConversationTurn] = deque(maxlen=window_size)
        self.total_turns = 0
        self.session_id: Optional[str] = None
        
        logger.info(f"初始化滑动窗口 Agent: window_size={window_size}")
    
    def add_conversation(self, human: str, ai: str) -> None:
        """添加对话到历史"""
        self.total_turns += 1
        turn = ConversationTurn(
            human=human,
            ai=ai,
            turn_number=self.total_turns,
        )
        self.conversation_history.append(turn)
        
        logger.debug(f"添加对话轮次 {self.total_turns} (窗口: {len(self.conversation_history)}/{self.window_size})")
    
    def get_context(self, query: str) -> Tuple[List[str], BaselineMetrics]:
        """
        获取当前上下文
        
        Args:
            query: 当前查询
        
        Returns:
            (上下文列表, 性能指标)
        """
        start_time = time.time()
        
        # 构建上下文（只包含窗口内的对话）
        context = []
        for turn in self.conversation_history:
            context.append(f"用户: {turn.human}\n助手: {turn.ai}")
        
        # 估算 token 数（简单估算：1 token ≈ 4 字符）
        context_text = "\n\n".join(context)
        context_length = len(context_text) // 4
        
        # 计算存储占用（估算）
        storage_bytes = len(context_text.encode('utf-8'))
        
        retrieval_latency = time.time() - start_time
        
        metrics = BaselineMetrics(
            total_turns=self.total_turns,
            context_length=context_length,
            storage_bytes=storage_bytes,
            retrieval_latency=retrieval_latency,
            early_info_accessible=(self.total_turns <= self.window_size),  # 只有早期轮次在窗口内时才能访问
        )
        
        return context, metrics
    
    def can_access_early_info(self, turn_number: int) -> bool:
        """
        检查是否能访问指定轮次的信息
        
        Args:
            turn_number: 轮次号（1-based）
        
        Returns:
            是否能访问
        """
        # 滑动窗口只能访问最近 window_size 轮
        return (self.total_turns - turn_number) < self.window_size
    
    def get_storage_size(self) -> int:
        """获取当前存储占用（字节）"""
        total_text = ""
        for turn in self.conversation_history:
            total_text += f"用户: {turn.human}\n助手: {turn.ai}\n\n"
        return len(total_text.encode('utf-8'))


class SummarizationAgent:
    """
    全量摘要 Agent（基线系统 2）
    
    维护完整对话历史，但当历史过长时，对早期对话进行摘要压缩。
    这是另一种常见的记忆管理策略。
    """
    
    def __init__(self, summary_threshold: int = 15, compression_ratio: float = 0.3):
        """
        初始化摘要 Agent
        
        Args:
            summary_threshold: 触发摘要的轮数阈值
            compression_ratio: 摘要压缩比（摘要保留原文本的比例）
        """
        self.summary_threshold = summary_threshold
        self.compression_ratio = compression_ratio
        self.conversation_history: List[ConversationTurn] = []
        self.summary: str = ""  # 压缩后的摘要
        self.total_turns = 0
        self.session_id: Optional[str] = None
        
        logger.info(
            f"初始化摘要 Agent: threshold={summary_threshold}, "
            f"compression_ratio={compression_ratio}"
        )
    
    def add_conversation(self, human: str, ai: str) -> None:
        """添加对话到历史"""
        self.total_turns += 1
        turn = ConversationTurn(
            human=human,
            ai=ai,
            turn_number=self.total_turns,
        )
        self.conversation_history.append(turn)
        
        # 如果超过阈值，触发摘要
        if len(self.conversation_history) > self.summary_threshold:
            self._create_summary()
        
        logger.debug(f"添加对话轮次 {self.total_turns} (历史: {len(self.conversation_history)} 轮)")
    
    def _create_summary(self) -> None:
        """创建摘要（简化版实现）"""
        # 简化的摘要：提取前 70% 的对话，压缩为关键信息
        # 实际应用中应该使用 LLM 进行摘要
        
        early_turns = self.conversation_history[:len(self.conversation_history) - self.summary_threshold]
        
        # 简单的文本压缩：保留关键信息
        summary_parts = []
        for turn in early_turns:
            # 提取关键信息（简化：取前50%的字符）
            human_summary = turn.human[:len(turn.human) // 2] + "..."
            ai_summary = turn.ai[:len(turn.ai) // 2] + "..."
            summary_parts.append(f"用户: {human_summary}\n助手: {ai_summary}")
        
        self.summary = "\n\n".join(summary_parts)
        
        # 保留最近 N 轮详细对话
        self.conversation_history = self.conversation_history[-self.summary_threshold:]
        
        logger.debug(f"创建摘要: {len(early_turns)} 轮对话被压缩，保留 {len(self.conversation_history)} 轮详细对话")
    
    def get_context(self, query: str) -> Tuple[List[str], BaselineMetrics]:
        """
        获取当前上下文
        
        Args:
            query: 当前查询
        
        Returns:
            (上下文列表, 性能指标)
        """
        start_time = time.time()
        
        # 构建上下文：摘要 + 最近详细对话
        context = []
        
        if self.summary:
            context.append(f"[历史摘要]\n{self.summary}")
        
        for turn in self.conversation_history:
            context.append(f"用户: {turn.human}\n助手: {turn.ai}")
        
        # 估算 token 数
        context_text = "\n\n".join(context)
        context_length = len(context_text) // 4
        
        # 计算存储占用
        storage_bytes = len(context_text.encode('utf-8'))
        # 加上摘要存储（虽然摘要已经压缩，但需要单独存储）
        storage_bytes += len(self.summary.encode('utf-8')) if self.summary else 0
        
        retrieval_latency = time.time() - start_time
        
        # 检查是否能访问早期信息（通过摘要）
        early_info_accessible = bool(self.summary) or (len(self.conversation_history) > 0)
        
        metrics = BaselineMetrics(
            total_turns=self.total_turns,
            context_length=context_length,
            storage_bytes=storage_bytes,
            retrieval_latency=retrieval_latency,
            early_info_accessible=early_info_accessible,
        )
        
        return context, metrics
    
    def can_access_early_info(self, turn_number: int) -> bool:
        """
        检查是否能访问指定轮次的信息
        
        Args:
            turn_number: 轮次号（1-based）
        
        Returns:
            是否能访问（摘要中包含该信息，或仍在详细对话中）
        """
        # 如果轮次在详细对话中
        if turn_number > (self.total_turns - len(self.conversation_history)):
            return True
        
        # 如果轮次在摘要中
        return bool(self.summary)
    
    def get_storage_size(self) -> int:
        """获取当前存储占用（字节）"""
        total_text = ""
        for turn in self.conversation_history:
            total_text += f"用户: {turn.human}\n助手: {turn.ai}\n\n"
        total_text += self.summary
        return len(total_text.encode('utf-8'))


class SimpleMockLLM:
    """
    简单的模拟 LLM
    
    用于基准测试，避免依赖真实的 LLM API。
    实际测试时可以用真实 LLM 替换。
    """
    
    def __init__(self, delay: float = 0.1):
        """
        初始化模拟 LLM
        
        Args:
            delay: 模拟延迟（秒）
        """
        self.delay = delay
    
    def generate(self, query: str, context: List[str]) -> Tuple[str, float]:
        """
        生成回答
        
        Args:
            query: 查询
            context: 上下文
        
        Returns:
            (生成的回答, 延迟时间)
        """
        start_time = time.time()
        
        # 模拟生成延迟
        time.sleep(self.delay)
        
        # 简单的响应生成（实际应该使用真实 LLM）
        context_summary = "\n".join(context[:2]) if context else "无上下文"
        response = f"基于上下文回答: {query}\n参考信息: {context_summary[:100]}..."
        
        latency = time.time() - start_time
        return response, latency


def create_baseline_agent(agent_type: str, **kwargs) -> Any:
    """
    创建基线 Agent
    
    Args:
        agent_type: Agent 类型 ("sliding_window" 或 "summarization")
        **kwargs: Agent 初始化参数
    
    Returns:
        Agent 实例
    """
    if agent_type == "sliding_window":
        window_size = kwargs.get("window_size", 10)
        return SlidingWindowAgent(window_size=window_size)
    elif agent_type == "summarization":
        threshold = kwargs.get("summary_threshold", 15)
        compression = kwargs.get("compression_ratio", 0.3)
        return SummarizationAgent(summary_threshold=threshold, compression_ratio=compression)
    else:
        raise ValueError(f"未知的 Agent 类型: {agent_type}")
