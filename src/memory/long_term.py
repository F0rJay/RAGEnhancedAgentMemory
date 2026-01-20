"""
长期记忆管理模块

实现长期记忆（Long-Term Semantic Memory）：
- 基于向量数据库的语义检索存储
- 支持异步归档与去重
- 从短期记忆归档内容到长期记忆
- 语义检索功能
"""

import time
import hashlib
import uuid
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        "sentence-transformers 未安装。请运行: pip install sentence-transformers>=2.3.0"
    ) from e

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from ..config import get_settings


class MemoryType(str, Enum):
    """记忆类型"""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    ARCHIVED = "archived"


@dataclass
class MemoryItem:
    """记忆项"""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    memory_type: MemoryType = MemoryType.LONG_TERM
    source_session: Optional[str] = None
    relevance_score: Optional[float] = None


class LongTermMemory:
    """
    长期记忆管理器
    
    负责将短期记忆归档到向量数据库，并提供语义检索功能。
    支持 Qdrant 和 Chroma 两种向量数据库后端。
    """

    def __init__(
        self,
        vector_db: Optional[str] = None,
        embedding_model: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_dim: Optional[int] = None,
    ):
        """
        初始化长期记忆管理器
        
        Args:
            vector_db: 向量数据库类型 ("qdrant" 或 "chroma")
            embedding_model: 嵌入模型名称
            collection_name: 集合名称
            embedding_dim: 向量维度
        """
        settings = get_settings()
        self.vector_db_type = vector_db or settings.vector_db
        self.embedding_model_name = embedding_model or settings.embedding_model
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.embedding_dim = embedding_dim or settings.embedding_dim

        # 初始化嵌入模型
        logger.info(f"加载嵌入模型: {self.embedding_model_name}")
        try:
            device = settings.embedding_device
            self.embedding_model = SentenceTransformer(
                self.embedding_model_name,
                device=device,
            )
            # 获取实际维度
            test_embedding = self.embedding_model.encode(["test"])
            self.embedding_dim = len(test_embedding[0])
            logger.info(f"嵌入模型加载成功，维度: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"嵌入模型加载失败: {e}")
            raise

        # 初始化向量数据库客户端
        self.client = None
        self.collection = None
        self._init_vector_db()

        # 去重缓存（内存中）
        self.content_hash_cache: Set[str] = set()

        logger.info(
            f"长期记忆管理器初始化完成: "
            f"vector_db={self.vector_db_type}, "
            f"collection={self.collection_name}"
        )

    def _init_vector_db(self) -> None:
        """初始化向量数据库客户端"""
        settings = get_settings()

        if self.vector_db_type == "qdrant":
            if not QDRANT_AVAILABLE:
                raise ImportError(
                    "Qdrant 客户端未安装。请运行: pip install qdrant-client>=1.7.0"
                )

            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )

            # 检查集合是否存在，不存在则创建
            try:
                collections = self.client.get_collections().collections
                collection_names = [c.name for c in collections]
                if self.collection_name not in collection_names:
                    logger.info(f"创建 Qdrant 集合: {self.collection_name}")
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(
                            size=self.embedding_dim,
                            distance=Distance.COSINE,
                        ),
                    )
            except Exception as e:
                logger.error(f"初始化 Qdrant 集合失败: {e}")
                raise

        elif self.vector_db_type == "chroma":
            if not CHROMA_AVAILABLE:
                raise ImportError(
                    "Chroma 客户端未安装。请运行: pip install chromadb>=0.4.0"
                )

            self.client = chromadb.Client(
                settings=ChromaSettings(
                    chroma_api_impl="rest",
                    chroma_server_host=settings.chroma_host,
                    chroma_server_http_port=settings.chroma_port,
                )
            )

            # 获取或创建集合
            try:
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as e:
                logger.error(f"初始化 Chroma 集合失败: {e}")
                raise
        else:
            raise ValueError(
                f"不支持的向量数据库类型: {self.vector_db_type}。"
                f"支持的类型: qdrant, chroma"
            )

    def _compute_content_hash(self, content: str) -> str:
        """
        计算内容的哈希值，用于去重
        
        Args:
            content: 文本内容
        
        Returns:
            MD5 哈希值
        """
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def _embed_text(self, text: str) -> List[float]:
        """
        将文本转换为向量
        
        Args:
            text: 文本内容
        
        Returns:
            向量表示
        """
        embedding = self.embedding_model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        批量将文本转换为向量
        
        Args:
            texts: 文本列表
        
        Returns:
            向量列表
        """
        embeddings = self.embedding_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return embeddings.tolist()

    def archive_from_short_term(
        self,
        conversation_turns: List[Dict[str, Any]],
        session_id: Optional[str] = None,
        deduplicate: bool = True,
    ) -> List[str]:
        """
        从短期记忆归档内容到长期记忆
        
        Args:
            conversation_turns: 短期记忆中的对话轮次列表
            session_id: 会话 ID
            deduplicate: 是否去重
        
        Returns:
            已归档的记忆项 ID 列表
        """
        archived_ids = []

        for turn in conversation_turns:
            # 提取文本内容
            human_msg = turn.get("human", turn.get("human_message", ""))
            ai_msg = turn.get("ai", turn.get("ai_message", ""))

            # 合并为完整内容
            content = f"用户: {human_msg}\n助手: {ai_msg}"

            # 去重检查
            if deduplicate:
                content_hash = self._compute_content_hash(content)
                if content_hash in self.content_hash_cache:
                    logger.debug(f"跳过重复内容: {content[:50]}...")
                    continue
                self.content_hash_cache.add(content_hash)

            # 归档
            memory_id = self.add_memory(
                content=content,
                metadata={
                    "source": "short_term_archive",
                    "session_id": session_id,
                    "timestamp": turn.get("timestamp", time.time()),
                },
                session_id=session_id,
            )
            archived_ids.append(memory_id)

        logger.info(f"从短期记忆归档了 {len(archived_ids)} 条内容")
        return archived_ids

    def add_memory(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        memory_id: Optional[str] = None,
    ) -> str:
        """
        添加记忆到长期记忆库
        
        Args:
            content: 记忆内容
            metadata: 元数据
            session_id: 会话 ID
            memory_id: 记忆 ID（如果不提供则自动生成）
        
        Returns:
            记忆 ID
        """
        if not memory_id:
            memory_id = str(uuid.uuid4())

        # 生成嵌入向量
        embedding = self._embed_text(content)

        # 准备元数据
        full_metadata = {
            "content": content,
            "timestamp": time.time(),
            "session_id": session_id,
            **(metadata or {}),
        }

        # 存储到向量数据库
        if self.vector_db_type == "qdrant":
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=memory_id,
                        vector=embedding,
                        payload=full_metadata,
                    )
                ],
            )
        elif self.vector_db_type == "chroma":
            self.collection.add(
                ids=[memory_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[full_metadata],
            )

        logger.debug(f"添加记忆到长期记忆库: {memory_id[:8]}...")
        return memory_id

    def add_memories_batch(
        self,
        contents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        session_id: Optional[str] = None,
    ) -> List[str]:
        """
        批量添加记忆
        
        Args:
            contents: 内容列表
            metadatas: 元数据列表
            session_id: 会话 ID
        
        Returns:
            记忆 ID 列表
        """
        if metadatas is None:
            metadatas = [{}] * len(contents)

        # 批量生成嵌入向量
        embeddings = self._embed_texts(contents)

        # 生成 ID
        memory_ids = [str(uuid.uuid4()) for _ in contents]

        # 准备数据
        if self.vector_db_type == "qdrant":
            points = []
            for i, (memory_id, content, embedding, metadata) in enumerate(
                zip(memory_ids, contents, embeddings, metadatas)
            ):
                full_metadata = {
                    "content": content,
                    "timestamp": time.time(),
                    "session_id": session_id,
                    **metadata,
                }
                points.append(
                    PointStruct(
                        id=memory_id,
                        vector=embedding,
                        payload=full_metadata,
                    )
                )

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

        elif self.vector_db_type == "chroma":
            full_metadatas = []
            for i, (content, metadata) in enumerate(zip(contents, metadatas)):
                full_metadata = {
                    "content": content,
                    "timestamp": time.time(),
                    "session_id": session_id,
                    **metadata,
                }
                full_metadatas.append(full_metadata)

            self.collection.add(
                ids=memory_ids,
                embeddings=embeddings,
                documents=contents,
                metadatas=full_metadatas,
            )

        logger.info(f"批量添加了 {len(memory_ids)} 条记忆")
        return memory_ids

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        session_id: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryItem]:
        """
        语义检索
        
        Args:
            query: 查询文本
            top_k: 返回前 K 个结果
            score_threshold: 相似度阈值
            session_id: 限制检索特定会话的记忆
            metadata_filter: 元数据过滤条件
        
        Returns:
            检索到的记忆项列表
        """
        # 生成查询向量
        query_embedding = self._embed_text(query)

        results = []

        if self.vector_db_type == "qdrant":
            # 构建过滤条件
            filter_conditions = []
            if session_id:
                filter_conditions.append(
                    FieldCondition(
                        key="session_id",
                        match=MatchValue(value=session_id),
                    )
                )
            if metadata_filter:
                for key, value in metadata_filter.items():
                    filter_conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value),
                        )
                    )

            search_filter = Filter(must=filter_conditions) if filter_conditions else None

            # 执行搜索
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=search_filter,
            )

            # 转换为 MemoryItem
            for result in search_results:
                memory_item = MemoryItem(
                    id=str(result.id),
                    content=result.payload.get("content", ""),
                    embedding=None,  # 不返回嵌入向量以节省内存
                    metadata=result.payload,
                    relevance_score=result.score,
                    memory_type=MemoryType.LONG_TERM,
                )
                results.append(memory_item)

        elif self.vector_db_type == "chroma":
            # 构建 where 条件
            where = {}
            if session_id:
                where["session_id"] = session_id
            if metadata_filter:
                where.update(metadata_filter)

            # 执行搜索
            search_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where if where else None,
            )

            # 转换为 MemoryItem
            if search_results["ids"] and len(search_results["ids"][0]) > 0:
                for i, memory_id in enumerate(search_results["ids"][0]):
                    distance = search_results["distances"][0][i] if "distances" in search_results else None
                    # Chroma 使用距离，转换为相似度分数（1 - distance for cosine）
                    score = 1.0 - distance if distance is not None else None

                    memory_item = MemoryItem(
                        id=memory_id,
                        content=search_results["documents"][0][i],
                        embedding=None,
                        metadata=search_results["metadatas"][0][i] if "metadatas" in search_results else {},
                        relevance_score=score,
                        memory_type=MemoryType.LONG_TERM,
                    )
                    results.append(memory_item)

        # 过滤低分结果
        if score_threshold is not None:
            results = [
                r for r in results
                if r.relevance_score is not None and r.relevance_score >= score_threshold
            ]

        logger.debug(f"检索到 {len(results)} 条记忆 (query: {query[:50]}...)")
        return results

    def delete_memory(self, memory_id: str) -> bool:
        """
        删除记忆
        
        Args:
            memory_id: 记忆 ID
        
        Returns:
            是否删除成功
        """
        try:
            if self.vector_db_type == "qdrant":
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=[memory_id],
                )
            elif self.vector_db_type == "chroma":
                self.collection.delete(ids=[memory_id])

            logger.debug(f"删除记忆: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"删除记忆失败: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """获取长期记忆统计信息"""
        stats = {
            "vector_db_type": self.vector_db_type,
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model_name,
            "embedding_dim": self.embedding_dim,
        }

        try:
            if self.vector_db_type == "qdrant":
                collection_info = self.client.get_collection(self.collection_name)
                stats["total_memories"] = collection_info.points_count
            elif self.vector_db_type == "chroma":
                stats["total_memories"] = self.collection.count()
        except Exception as e:
            logger.warning(f"获取统计信息失败: {e}")
            stats["total_memories"] = "unknown"

        return stats
