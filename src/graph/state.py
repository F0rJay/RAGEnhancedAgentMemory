"""
LangGraph 状态定义

定义 Agent 的状态结构，包括瞬时记忆和中间处理状态。
基于 LangGraph 的状态管理模式，支持状态持久化和检查点恢复。
"""

from typing import TypedDict, List, Annotated, Optional, Dict, Any
from typing_extensions import NotRequired
from langchain_core.messages import BaseMessage
from operator import add


class AgentState(TypedDict):
    """
    Agent 状态定义
    
    该状态对象在整个 Agent 执行过程中传递，包含：
    - 瞬时记忆：当前请求的输入和中间结果
    - 检索结果：从长期记忆检索到的文档
    - 生成结果：LLM 生成的回答
    - 质量评估：相关性评分和幻觉检测结果
    - 控制流：重试计数和路由决策
    """

    # ========== 输入与上下文 ==========
    input: str
    """用户输入的问题或指令"""

    chat_history: Annotated[List[BaseMessage], add]
    """
    会话历史记录
    
    使用 Annotated 和 add reducer 来合并多个节点产生的消息。
    每个节点可以添加新的消息，LangGraph 会自动合并。
    """

    # ========== 检索增强结果 ==========
    documents: Annotated[List[str], add]
    """
    检索到的文档列表
    
    从长期记忆或短期记忆中检索到的相关文档片段。
    多个检索节点可以添加文档，系统会自动合并并去重。
    """

    document_metadata: Annotated[List[Dict[str, Any]], add]
    """
    文档元数据列表
    
    对应 documents 中每个文档的元数据，包括：
    - source: 文档来源
    - timestamp: 归档时间
    - relevance_score: 相关性分数
    - memory_type: 记忆类型 (short-term/long-term)
    """

    # ========== 生成与评估 ==========
    generation: str
    """LLM 生成的最终回答"""

    relevance_score: str
    """
    文档相关性评分
    
    可能的取值：
    - "yes": 检索到的文档高度相关
    - "no": 检索到的文档不相关，需要重新检索
    - "partial": 部分相关，可能需要补充检索
    """

    hallucination_score: str
    """
    幻觉检测评分
    
    可能的取值：
    - "grounded": 生成内容基于检索到的文档，无幻觉
    - "not_grounded": 生成内容包含幻觉或未基于文档
    - "uncertain": 无法确定，需要人工审核
    """

    # ========== 控制流 ==========
    retry_count: int
    """
    重试计数器
    
    用于防止无限循环，当重试次数超过阈值时终止执行。
    每次失败的重试都会增加此计数。
    """

    should_continue: NotRequired[bool]
    """
    是否继续执行标志
    
    用于控制图的执行流程，决定是否需要进一步检索或生成。
    """

    # ========== 记忆路由信息 ==========
    memory_type: NotRequired[str]
    """
    使用的记忆类型
    
    可能的值：
    - "short_term": 使用短期记忆
    - "long_term": 使用长期记忆
    - "hybrid": 混合使用两种记忆
    """

    retrieval_query: NotRequired[str]
    """
    检索查询文本
    
    用于从向量数据库检索的查询，可能与原始 input 不同
    （例如：经过查询改写或扩展）
    """

    # ========== 错误处理 ==========
    error: NotRequired[Optional[str]]
    """
    错误信息
    
    如果执行过程中发生错误，存储错误信息以便后续处理。
    """

    # ========== 性能指标 ==========
    retrieval_latency: NotRequired[float]
    """
    检索延迟（秒）
    
    记录从长期记忆检索文档所花费的时间。
    """

    generation_latency: NotRequired[float]
    """
    生成延迟（秒）
    
    记录 LLM 生成回答所花费的时间。
    """


def reduce_chat_history(left: List[BaseMessage], right: List[BaseMessage]) -> List[BaseMessage]:
    """
    自定义聊天历史合并函数
    
    除了简单的列表合并，还可以添加去重逻辑。
    
    Args:
        left: 已有的消息列表
        right: 新添加的消息列表
    
    Returns:
        合并后的消息列表
    """
    # 简单合并（LangGraph 默认使用 add，效果相同）
    # 这里可以添加自定义逻辑，比如去重、排序等
    combined = left + right
    # 去重：基于消息内容和类型
    seen = set()
    unique_messages = []
    for msg in combined:
        # 使用消息的字符串表示作为唯一标识
        msg_key = (type(msg).__name__, msg.content[:100] if hasattr(msg, 'content') else str(msg))
        if msg_key not in seen:
            seen.add(msg_key)
            unique_messages.append(msg)
    return unique_messages


def reduce_documents(left: List[str], right: List[str]) -> List[str]:
    """
    自定义文档合并函数
    
    合并文档列表并去除重复项。
    
    Args:
        left: 已有的文档列表
        right: 新添加的文档列表
    
    Returns:
        合并并去重后的文档列表
    """
    # 合并列表
    combined = left + right
    # 去重（保留顺序）
    seen = set()
    unique_docs = []
    for doc in combined:
        # 使用文档内容的哈希作为唯一标识
        doc_hash = hash(doc[:500])  # 只考虑前500字符，提高性能
        if doc_hash not in seen:
            seen.add(doc_hash)
            unique_docs.append(doc)
    return unique_docs


# 使用自定义 reducer 的状态定义（可选，如果需要自定义合并逻辑）
class AgentStateWithCustomReducer(TypedDict):
    """
    带自定义 reducer 的 Agent 状态定义
    
    如果需要更复杂的合并逻辑，可以使用此定义。
    """
    input: str
    chat_history: Annotated[List[BaseMessage], reduce_chat_history]
    documents: Annotated[List[str], reduce_documents]
    document_metadata: Annotated[List[Dict[str, Any]], add]
    generation: str
    relevance_score: str
    hallucination_score: str
    retry_count: int
    should_continue: NotRequired[bool]
    memory_type: NotRequired[str]
    retrieval_query: NotRequired[str]
    error: NotRequired[Optional[str]]
    retrieval_latency: NotRequired[float]
    generation_latency: NotRequired[float]


# 导出默认使用的状态定义
__all__ = [
    "AgentState",
    "AgentStateWithCustomReducer",
    "reduce_chat_history",
    "reduce_documents",
]
