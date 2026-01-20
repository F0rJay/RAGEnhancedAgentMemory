"""
RAG 增强型 Agent 记忆系统核心模块

整合所有子模块，提供统一的接口供上层应用使用。
"""

import time
import uuid
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path

try:
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
except ImportError as e:
    raise ImportError(
        "langchain-core 未安装。请运行: pip install langchain-core>=0.3.0"
    ) from e

try:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_CHECKPOINT_AVAILABLE = True
except ImportError:
    LANGGRAPH_CHECKPOINT_AVAILABLE = False
    BaseCheckpointSaver = None
    MemorySaver = None

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from .config import get_settings
from .memory.short_term import ShortTermMemory
from .memory.long_term import LongTermMemory
from .memory.routing import MemoryRouter, RoutingContext, RouteDecision
from .retrieval.hybrid import HybridRetriever
from .retrieval.reranker import Reranker
from .graph.state import AgentState


class RAGEnhancedAgentMemory:
    """
    RAG 增强型 Agent 记忆系统
    
    整合所有子模块，提供统一的记忆管理接口：
    - 短期记忆管理
    - 长期记忆归档与检索
    - 自适应路由决策
    - 混合检索与重排序
    - LangGraph 集成
    """

    def __init__(
        self,
        vector_db: Optional[str] = None,
        embedding_model: Optional[str] = None,
        rerank_model: Optional[str] = None,
        short_term_threshold: Optional[int] = None,
        long_term_trigger: Optional[float] = None,
        use_hybrid_retrieval: bool = True,
        use_rerank: bool = True,
        checkpoint_dir: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        """
        初始化 RAG 增强型 Agent 记忆系统
        
        Args:
            vector_db: 向量数据库类型 ("qdrant" 或 "chroma")
            embedding_model: 嵌入模型名称
            rerank_model: 重排序模型名称
            short_term_threshold: 短期记忆轮数阈值
            long_term_trigger: 长期记忆触发阈值
            use_hybrid_retrieval: 是否使用混合检索
            use_rerank: 是否使用重排序
            checkpoint_dir: 检查点目录
            session_id: 会话 ID
        """
        settings = get_settings()

        # 配置参数
        self.vector_db = vector_db or settings.vector_db
        self.embedding_model = embedding_model or settings.embedding_model
        self.rerank_model = rerank_model or settings.rerank_model
        self.short_term_threshold = short_term_threshold or settings.short_term_threshold
        self.long_term_trigger = long_term_trigger or settings.long_term_trigger
        self.use_hybrid_retrieval = use_hybrid_retrieval
        self.use_rerank = use_rerank
        self.checkpoint_dir = checkpoint_dir or settings.checkpoint_dir
        self.session_id = session_id or str(uuid.uuid4())

        # 初始化子模块
        logger.info("初始化 RAG 增强型 Agent 记忆系统...")

        # 短期记忆
        self.short_term_memory = ShortTermMemory(
            max_turns=self.short_term_threshold,
        )
        self.short_term_memory.set_session_id(self.session_id)

        # 长期记忆
        self.long_term_memory = LongTermMemory(
            vector_db=self.vector_db,
            embedding_model=self.embedding_model,
        )

        # 重排序器（可选）
        self.reranker = None
        if self.use_rerank:
            try:
                self.reranker = Reranker(model_name=self.rerank_model)
            except Exception as e:
                logger.warning(f"重排序器初始化失败，将不使用重排序: {e}")

        # 混合检索器（可选）
        self.hybrid_retriever = None
        if self.use_hybrid_retrieval:
            self.hybrid_retriever = HybridRetriever(
                long_term_memory=self.long_term_memory,
                reranker=self.reranker if self.use_rerank else None,
                use_rerank=self.use_rerank,
            )

        # 记忆路由器
        self.router = MemoryRouter(
            short_term_threshold=self.short_term_threshold,
            long_term_trigger=self.long_term_trigger,
        )

        # 检查点保存器（用于 LangGraph）
        self.checkpointer = None
        if LANGGRAPH_CHECKPOINT_AVAILABLE:
            try:
                checkpoint_path = Path(self.checkpoint_dir)
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                # 使用内存检查点（简化版，实际可用文件系统检查点）
                self.checkpointer = MemorySaver()
                logger.info(f"检查点系统初始化: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"检查点系统初始化失败: {e}")

        logger.info(
            f"RAG 增强型 Agent 记忆系统初始化完成: "
            f"session_id={self.session_id}, "
            f"vector_db={self.vector_db}, "
            f"use_hybrid={self.use_hybrid_retrieval}, "
            f"use_rerank={self.use_rerank}"
        )

    def retrieve_context(
        self,
        state: AgentState,
    ) -> Dict[str, Any]:
        """
        检索上下文（用于 LangGraph 节点）
        
        Args:
            state: Agent 状态
        
        Returns:
            更新的状态字典
        """
        query = state.get("input", "")
        if not query:
            return {"documents": [], "memory_type": "none"}

        # 创建路由上下文
        routing_context = RoutingContext(
            query=query,
            short_term_memory=self.short_term_memory,
            long_term_memory=self.long_term_memory,
            session_id=self.session_id,
            recent_relevance_score=state.get("relevance_score"),
        )

        # 执行路由决策
        routing_result = self.router.route(routing_context)

        # 准备检索结果
        documents = []
        document_metadata = []
        memory_type = routing_result.decision.value

        # 添加短期记忆上下文
        if routing_result.short_term_context:
            short_term_turns = routing_result.short_term_context.get("conversation", [])
            for turn in short_term_turns:
                context_text = f"用户: {turn.get('human', '')}\n助手: {turn.get('ai', '')}"
                documents.append(context_text)
                document_metadata.append({
                    "source": "short_term",
                    "timestamp": turn.get("timestamp", time.time()),
                })

        # 添加长期记忆检索结果
        if routing_result.long_term_results:
            for item in routing_result.long_term_results:
                documents.append(item.content)
                document_metadata.append({
                    "source": "long_term",
                    "score": item.relevance_score,
                    **item.metadata,
                })

        # 如果使用了混合检索，使用混合检索器
        if self.use_hybrid_retrieval and self.hybrid_retriever and not routing_result.long_term_results:
            try:
                hybrid_results = self.hybrid_retriever.retrieve(
                    query=query,
                    top_k=5,
                    session_id=self.session_id,
                )
                for result in hybrid_results:
                    documents.append(result.content)
                    document_metadata.append({
                        "source": result.source,
                        "vector_score": result.vector_score,
                        "keyword_score": result.keyword_score,
                        "combined_score": result.combined_score,
                        **result.metadata,
                    })
                memory_type = "hybrid"
            except Exception as e:
                logger.warning(f"混合检索失败: {e}")

        return {
            "documents": documents,
            "document_metadata": document_metadata,
            "memory_type": memory_type,
            "retrieval_query": query,
        }

    def save_context(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, str],
    ) -> None:
        """
        保存上下文到记忆系统
        
        Args:
            inputs: 输入字典（包含 input）
            outputs: 输出字典（包含 generation）
        """
        user_input = inputs.get("input", "")
        ai_response = outputs.get("generation", "")

        if user_input and ai_response:
            # 保存到短期记忆
            self.short_term_memory.add_conversation_turn(
                human_message=user_input,
                ai_message=ai_response,
                metadata={
                    "timestamp": time.time(),
                    "session_id": self.session_id,
                },
            )

            logger.debug(f"保存对话到短期记忆: {user_input[:50]}...")

    def get_checkpointer(self) -> Optional[BaseCheckpointSaver]:
        """
        获取 LangGraph 检查点保存器
        
        Returns:
            检查点保存器实例
        """
        return self.checkpointer

    def add_memory(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        添加记忆到长期记忆库
        
        Args:
            content: 记忆内容
            metadata: 元数据
        
        Returns:
            记忆 ID
        """
        return self.long_term_memory.add_memory(
            content=content,
            metadata=metadata,
            session_id=self.session_id,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        use_hybrid: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        搜索记忆
        
        Args:
            query: 查询文本
            top_k: 返回 Top-K 结果
            use_hybrid: 是否使用混合检索（None 表示使用初始化时的设置）
        
        Returns:
            搜索结果列表
        """
        use_hybrid = use_hybrid if use_hybrid is not None else self.use_hybrid_retrieval

        if use_hybrid and self.hybrid_retriever:
            # 使用混合检索
            results = self.hybrid_retriever.retrieve(
                query=query,
                top_k=top_k,
                session_id=self.session_id,
            )
            return [
                {
                    "content": r.content,
                    "score": r.combined_score,
                    "source": r.source,
                    "metadata": r.metadata,
                }
                for r in results
            ]
        else:
            # 使用纯向量检索
            results = self.long_term_memory.search(
                query=query,
                top_k=top_k,
                session_id=self.session_id,
            )
            return [
                {
                    "content": item.content,
                    "score": item.relevance_score,
                    "metadata": item.metadata,
                }
                for item in results
            ]

    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        return {
            "session_id": self.session_id,
            "vector_db": self.vector_db,
            "short_term": self.short_term_memory.get_stats(),
            "long_term": self.long_term_memory.get_stats(),
            "use_hybrid_retrieval": self.use_hybrid_retrieval,
            "use_rerank": self.use_rerank,
        }

    def clear_short_term_memory(self) -> None:
        """清空短期记忆"""
        self.short_term_memory.clear()
        logger.info("短期记忆已清空")

    def archive_short_term_to_long_term(self) -> List[str]:
        """
        将短期记忆归档到长期记忆
        
        Returns:
            已归档的记忆 ID 列表
        """
        context = self.short_term_memory.get_recent_context(include_entities=False)
        conversation_turns = context.get("conversation", [])

        if not conversation_turns:
            return []

        archived_ids = self.long_term_memory.archive_from_short_term(
            conversation_turns=conversation_turns,
            session_id=self.session_id,
            deduplicate=True,
        )

        logger.info(f"归档了 {len(archived_ids)} 条记忆到长期记忆库")
        return archived_ids

    def get_context_for_llm(self, max_tokens: Optional[int] = None) -> str:
        """
        获取格式化的上下文字符串，用于 LLM 输入
        
        Args:
            max_tokens: 最大 token 数
        
        Returns:
            格式化的上下文字符串
        """
        return self.short_term_memory.get_context_for_llm(max_tokens=max_tokens)
