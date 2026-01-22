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
import math
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # 使用 math 作为后备
    np = None

# 延迟导入 SentenceTransformer，避免模块级导入触发 torch
# SentenceTransformer 将在 LongTermMemory.__init__ 中按需导入
SentenceTransformer = None  # 占位符，实际导入在 __init__ 方法中

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
    from chromadb import PersistentClient
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    PersistentClient = None
    chromadb = None

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

        # 初始化嵌入模型（延迟导入，避免模块级导入触发 torch）
        logger.info(f"加载嵌入模型: {self.embedding_model_name}")
        try:
            device = settings.embedding_device
            # 延迟导入 SentenceTransformer，避免模块级导入触发 torch
            # 这样可以确保用户 import 插件时不会触发 torch 导入
            from sentence_transformers import SentenceTransformer
            
            # 尝试离线加载，如果本地有模型文件就不需要联网
            import os
            # 设置离线模式环境变量（如果还没有设置）
            if "TRANSFORMERS_OFFLINE" not in os.environ:
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
            if "HF_DATASETS_OFFLINE" not in os.environ:
                os.environ["HF_DATASETS_OFFLINE"] = "1"
            
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
            # 如果离线模式失败，尝试清除离线标志后重试（可能需要下载）
            import os
            if os.environ.get("TRANSFORMERS_OFFLINE") == "1":
                logger.warning("离线模式加载失败，尝试允许联网下载...")
                del os.environ["TRANSFORMERS_OFFLINE"]
                del os.environ["HF_DATASETS_OFFLINE"]
                try:
                    # 延迟导入 SentenceTransformer
                    from sentence_transformers import SentenceTransformer
                    self.embedding_model = SentenceTransformer(
                        self.embedding_model_name,
                        device=device,
                    )
                    test_embedding = self.embedding_model.encode(["test"])
                    self.embedding_dim = len(test_embedding[0])
                    logger.info(f"嵌入模型加载成功（联网模式），维度: {self.embedding_dim}")
                except Exception as e2:
                    logger.error(f"联网模式下嵌入模型加载也失败: {e2}")
                    raise
            else:
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
    
    def close(self) -> None:
        """
        关闭向量数据库客户端连接
        
        应该在程序退出前调用，以避免退出时的清理警告
        """
        try:
            if self.client is not None:
                if self.vector_db_type == "qdrant" and hasattr(self.client, 'close'):
                    try:
                        self.client.close()
                        logger.debug("Qdrant 客户端已关闭")
                    except Exception as e:
                        logger.debug(f"关闭 Qdrant 客户端时出错（可忽略）: {e}")
                elif self.vector_db_type == "chroma" and hasattr(self.client, 'persist'):
                    # Chroma 客户端通常不需要显式关闭，但可以调用 persist 确保数据保存
                    try:
                        self.client.persist()
                        logger.debug("Chroma 客户端数据已持久化")
                    except Exception:
                        pass
        except Exception as e:
            logger.debug(f"清理长期记忆资源时出错（可忽略）: {e}")
    
    def __del__(self):
        """析构函数，尝试清理资源"""
        try:
            self.close()
        except Exception:
            pass  # 忽略所有错误，避免在 Python 关闭时出错

    def _init_vector_db(self) -> None:
        """初始化向量数据库客户端"""
        settings = get_settings()

        if self.vector_db_type == "qdrant":
            if not QDRANT_AVAILABLE:
                raise ImportError(
                    "Qdrant 客户端未安装。请运行: pip install qdrant-client>=1.7.0"
                )

            # 使用本地嵌入式模式（不需要服务器或Docker）
            # 优先使用本地模式，避免需要 Docker
            try:
                from pathlib import Path
                # 使用项目根目录下的 qdrant_db 文件夹
                qdrant_db_path = Path("./qdrant_db").absolute()
                qdrant_db_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"使用 Qdrant 本地嵌入式模式，路径: {qdrant_db_path}")
                self.client = QdrantClient(path=str(qdrant_db_path))
            except Exception as e:
                # 如果本地嵌入式模式失败，尝试使用远程服务器模式
                logger.warning(f"使用 Qdrant 本地嵌入式模式失败，尝试远程服务器模式: {e}")
                try:
                    self.client = QdrantClient(
                        url=settings.qdrant_url,
                        api_key=settings.qdrant_api_key,
                    )
                except Exception as e2:
                    logger.error(f"Qdrant 远程服务器模式也失败: {e2}")
                    raise

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

            # 使用本地持久化模式（不需要服务器）
            # 优先使用本地模式，避免需要 Docker
            try:
                from pathlib import Path
                # 使用项目根目录下的 chroma_db 文件夹
                chroma_db_path = Path("./chroma_db").absolute()
                chroma_db_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"使用 Chroma 本地持久化模式，路径: {chroma_db_path}")
                self.client = PersistentClient(path=str(chroma_db_path))
            except Exception as e:
                # 如果本地模式失败，尝试 REST API 模式（需要服务器）
                logger.warning(f"使用 Chroma 本地模式失败，尝试 REST API 模式: {e}")
                try:
                    self.client = chromadb.Client(
                        settings=ChromaSettings(
                            chroma_api_impl="rest",
                            chroma_server_host=settings.chroma_host,
                            chroma_server_http_port=settings.chroma_port,
                        )
                    )
                except Exception as e2:
                    logger.error(f"Chroma REST API 模式也失败: {e2}")
                    raise

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

    def _check_semantic_similarity(
        self,
        new_content: str,
        existing_contents: List[str],
        threshold: float = 0.95,
    ) -> bool:
        """
        检查新内容是否与已有内容语义相似
        
        Args:
            new_content: 新内容
            existing_contents: 已有内容列表
            threshold: 相似度阈值 (0-1)
        
        Returns:
            如果存在相似内容返回 True，否则返回 False
        """
        if not existing_contents:
            return False
        
        try:
            # 批量计算嵌入向量
            all_contents = [new_content] + existing_contents
            embeddings = self._embed_texts(all_contents)
            new_embedding = embeddings[0]
            existing_embeddings = embeddings[1:]
            
            # 计算与新内容的相似度
            if not NUMPY_AVAILABLE:
                logger.warning("NumPy 未安装，无法进行语义相似度检查")
                return False
            
            new_embedding_np = np.array(new_embedding)
            for existing_embedding in existing_embeddings:
                existing_embedding_np = np.array(existing_embedding)
                # 余弦相似度
                similarity = np.dot(new_embedding_np, existing_embedding_np) / (
                    np.linalg.norm(new_embedding_np) * np.linalg.norm(existing_embedding_np)
                )
                
                if similarity >= threshold:
                    logger.debug(f"发现语义相似内容，相似度: {similarity:.4f} >= {threshold}")
                    return True
            
            return False
        except Exception as e:
            logger.warning(f"语义相似度检查失败: {e}，使用哈希去重")
            return False

    def _get_recent_contents(self, limit: int = 100) -> List[str]:
        """
        获取最近的内容列表（用于语义相似度检查）
        
        Args:
            limit: 获取最近多少条内容
        
        Returns:
            内容列表
        """
        try:
            if self.vector_db_type == "qdrant":
                # 获取最近的 points（通过时间戳排序）
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=limit,
                    with_payload=True,
                )
                contents = [
                    point.payload.get("content", "")
                    for point in scroll_result[0]
                    if point.payload.get("content")
                ]
                return contents
            elif self.vector_db_type == "chroma":
                # Chroma 的 get 实现
                results = self.collection.get(limit=limit)
                return results.get("documents", []) if results.get("documents") else []
        except Exception as e:
            logger.warning(f"获取最近内容失败: {e}")
            return []
        
        return []

    def archive_from_short_term(
        self,
        conversation_turns: List[Dict[str, Any]],
        session_id: Optional[str] = None,
        deduplicate: bool = True,
        semantic_dedup: bool = True,
        filter_low_value: bool = False,
    ) -> List[str]:
        """
        从短期记忆归档内容到长期记忆
        
        Args:
            conversation_turns: 短期记忆中的对话轮次列表
            session_id: 会话 ID
            deduplicate: 是否去重（基于哈希）
            semantic_dedup: 是否使用语义相似度去重
            filter_low_value: 是否过滤低价值信息
        
        Returns:
            已归档的记忆项 ID 列表
        """
        settings = get_settings()
        archived_ids = []
        semantic_dedup_threshold = settings.semantic_dedup_threshold

        # 如果启用语义去重，先获取最近的内容（策略B：时间窗口约束）
        recent_contents = []
        if semantic_dedup:
            # 策略B：只检查当前Session的最近内容，缩小时间窗口避免过度去重
            # 优先获取当前Session的最近内容（2-3分钟或最近3-5轮）
            try:
                if self.vector_db_type == "qdrant":
                    scroll_result = self.client.scroll(
                        collection_name=self.collection_name,
                        limit=20,  # 限制最近20条，降低跨Session干扰
                        with_payload=True,
                    )
                    # 过滤出当前Session或最近2-3分钟内的内容
                    current_time = time.time()
                    session_points = [
                        point for point in scroll_result[0]
                        if point.payload.get("session_id") == session_id or 
                        (current_time - point.payload.get("timestamp", 0)) < 180  # 最近3分钟内的内容（缩小窗口）
                    ]
                    recent_contents = [
                        point.payload.get("content", "")
                        for point in session_points[:10]  # 最多10条用于去重检查（缩小范围）
                        if point.payload.get("content")
                    ]
                elif self.vector_db_type == "chroma":
                    # Chroma 类似处理
                    results = self.collection.get(limit=20)
                    if "documents" in results:
                        # 只取最近10条
                        recent_contents = results["documents"][:10]
            except Exception as e:
                logger.warning(f"获取最近内容失败，使用空列表: {e}")
                recent_contents = []

        # 低价值关键词
        low_value_keywords = {
            "你好", "谢谢", "好的", "在吗", "收到", "嗯", "OK", "明白", "知道了",
            "哦", "嗯嗯", "好的好的", "谢谢谢谢", "不客气", "没问题", "行", "可以",
            "了解", "清楚", "明白了", "收到", "嗯",
        }
        
        # 实体白名单关键词（包含这些词的内容强制保留）
        entity_whitelist_keywords = {
            "码", "色", "号", "钱", "元", "块", "kg", "斤", "地址", "手机",
            "尺码", "颜色", "订单号", "ORDER", "价格",
        }
        
        # 意图白名单关键词（业务动作词，决定是否为业务指令）
        intent_whitelist_keywords = {
            # 购买意图
            "买", "购", "下单", "定", "要", "需要",
            # 售后意图
            "退", "换", "修", "坏", "错", "不合适", "问题",
            # 咨询意图
            "查", "问", "哪里", "什么", "怎么", "如何", "能否", "可以",
            # 物流意图
            "物流", "快递", "发货", "配送", "到", "送达", "派送",
            # 资产/凭证
            "单", "票", "券", "会员", "订单",
            # 客服/权益意图
            "电话", "人工", "联系", "包邮", "优惠", "折扣", "客服",
        }
        
        # 合并所有白名单关键词
        all_whitelist_keywords = entity_whitelist_keywords | intent_whitelist_keywords

        for turn in conversation_turns:
            # 提取文本内容
            human_msg = turn.get("human", turn.get("human_message", ""))
            ai_msg = turn.get("ai", turn.get("ai_message", ""))
            
            # 低价值过滤（只过滤用户和助手都是低价值的情况）
            if filter_low_value:
                human_msg_clean = human_msg.strip()
                ai_msg_clean = ai_msg.strip()
                
                # 策略A：实体白名单 + 意图白名单检查 - 包含数字或业务关键词的强制保留
                import re
                combined_text = human_msg_clean + ai_msg_clean
                has_entity_or_intent = False
                # 检查是否包含数字（订单号、价格等）
                if re.search(r'\d+', combined_text):
                    has_entity_or_intent = True
                # 检查是否包含白名单关键词（实体或意图）
                if any(keyword in combined_text for keyword in all_whitelist_keywords):
                    has_entity_or_intent = True
                
                # 如果有实体或意图信息，强制保留（即使是短句）
                if has_entity_or_intent:
                    logger.debug(f"保留业务指令对话: {human_msg[:50]}...")
                    # 继续处理，不跳过
                else:
                    # 原有低价值检查逻辑
                    human_is_low = (human_msg_clean in low_value_keywords or len(human_msg_clean) <= 2)
                    ai_is_low = (ai_msg_clean in low_value_keywords or len(ai_msg_clean) <= 2)
                    
                    # 只有当用户和助手都是低价值时才过滤
                    if human_is_low and ai_is_low:
                        logger.debug(f"过滤低价值对话: {human_msg[:50]}...")
                        continue
                    
                    # 检查信息熵（只对纯低价值消息应用）
                    if human_is_low and len(set(human_msg_clean)) < 3 and len(human_msg_clean) < 5:
                        continue

            # 合并为完整内容
            content = f"用户: {human_msg}\n助手: {ai_msg}"

            # 哈希去重检查
            if deduplicate:
                content_hash = self._compute_content_hash(content)
                if content_hash in self.content_hash_cache:
                    logger.debug(f"跳过重复内容（哈希）: {content[:50]}...")
                    continue
                self.content_hash_cache.add(content_hash)

            # 语义相似度去重检查（如果相似，强化原有记忆而非跳过）
            if semantic_dedup and recent_contents:
                if self._check_semantic_similarity(content, recent_contents, semantic_dedup_threshold):
                    # 尝试找到并强化最相似的原有记忆
                    try:
                        if self.vector_db_type == "qdrant":
                            # 使用向量搜索找到最相似的记忆
                            content_embedding = self._embed_text(content)
                            from qdrant_client.models import NearestQuery
                            nearest_query = NearestQuery(nearest=content_embedding)
                            
                            query_response = self.client.query_points(
                                collection_name=self.collection_name,
                                query=nearest_query,
                                limit=1,  # 只找最相似的1条
                                score_threshold=semantic_dedup_threshold,
                            )
                            
                            if query_response.points and len(query_response.points) > 0:
                                # 找到相似记忆，更新其访问计数
                                similar_point = query_response.points[0]
                                current_count = similar_point.payload.get("access_count", 0)
                                new_count = current_count + 1
                                
                                self.client.set_payload(
                                    collection_name=self.collection_name,
                                    payload={
                                        "access_count": new_count,
                                        "last_accessed": time.time(),
                                    },
                                    points=[similar_point.id],
                                )
                                logger.debug(f"强化相似记忆（ID: {similar_point.id[:8]}...），访问计数: {current_count} -> {new_count}")
                        # Chroma 类似处理
                        elif self.vector_db_type == "chroma":
                            content_embedding = self._embed_text(content)
                            results = self.collection.query(
                                query_embeddings=[content_embedding],
                                n_results=1,
                                where={} if not session_id else {"session_id": session_id},
                            )
                            if results["ids"] and len(results["ids"][0]) > 0:
                                memory_id = results["ids"][0][0]
                                current_metadata = results["metadatas"][0][0] if results["metadatas"] else {}
                                current_count = current_metadata.get("access_count", 0)
                                new_count = current_count + 1
                                
                                self.collection.update(
                                    ids=[memory_id],
                                    metadatas=[{**current_metadata, "access_count": new_count, "last_accessed": time.time()}],
                                )
                                logger.debug(f"强化相似记忆（ID: {memory_id[:8]}...），访问计数: {current_count} -> {new_count}")
                    except Exception as e:
                        logger.warning(f"强化相似记忆失败: {e}")
                    
                    logger.debug(f"跳过重复内容（语义相似，已强化原有记忆）: {content[:50]}...")
                    continue

            # 归档
            memory_id = self.add_memory(
                content=content,
                metadata={
                    "source": "short_term_archive",
                    "session_id": session_id,
                    "timestamp": turn.get("timestamp", time.time()),
                    "access_count": 0,  # 初始化访问计数
                },
                session_id=session_id,
            )
            archived_ids.append(memory_id)
            
            # 更新最近内容列表（用于后续相似度检查）
            if semantic_dedup:
                recent_contents.append(content)

        logger.info(f"从短期记忆归档了 {len(archived_ids)} 条内容（去重后）")
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
        # 先设置默认值，然后允许 metadata 覆盖
        default_metadata = {
            "content": content,
            "timestamp": time.time(),
            "session_id": session_id,
            "access_count": 0,  # 初始化访问计数
        }
        
        # 合并用户提供的元数据（允许覆盖默认值）
        if metadata:
            full_metadata = {**default_metadata, **metadata}
        else:
            full_metadata = default_metadata

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
                    "access_count": 0,  # 初始化访问计数
                    **metadata,  # 允许覆盖
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
                    "access_count": 0,  # 初始化访问计数
                    **metadata,  # 允许覆盖
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

            # 执行搜索 - 使用 query_points API（新版本 qdrant-client）
            from qdrant_client.models import NearestQuery
            nearest_query = NearestQuery(nearest=query_embedding)
            
            query_response = self.client.query_points(
                collection_name=self.collection_name,
                query=nearest_query,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=search_filter,
            )
            search_results = query_response.points

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

        # 应用时间加权衰减和访问计数强化
        settings = get_settings()
        if settings.enable_time_decay:
            current_time = time.time()
            lambda_param = settings.time_decay_lambda
            alpha = settings.access_count_alpha
            
            for result in results:
                original_score = result.relevance_score
                if original_score is None:
                    continue
                
                # 获取时间戳和访问计数
                metadata = result.metadata
                timestamp = metadata.get("timestamp", current_time)
                access_count = metadata.get("access_count", 0)
                
                # 计算时间衰减（年龄，单位：秒）
                age_seconds = current_time - timestamp
                
                # 时间衰减公式: e^(-λ * t)
                time_decay = math.exp(-lambda_param * age_seconds)
                
                # 访问计数强化: 1 + α * log(1 + access_count)
                reinforcement = 1.0 + alpha * math.log(1.0 + access_count)
                
                # 综合分数
                adjusted_score = original_score * time_decay * reinforcement
                result.relevance_score = float(adjusted_score)
                
                # 更新访问计数（在元数据中，但实际更新需要在数据库中）
                # 这里先更新元数据，实际更新需要在返回前完成
                result.metadata["access_count"] = access_count + 1
            
            # 按调整后的分数重新排序
            results.sort(key=lambda x: x.relevance_score or 0.0, reverse=True)
            
            # 更新数据库中的访问计数
            self._update_access_counts([r.id for r in results])

        # 过滤低分结果
        if score_threshold is not None:
            results = [
                r for r in results
                if r.relevance_score is not None and r.relevance_score >= score_threshold
            ]

        logger.debug(f"检索到 {len(results)} 条记忆 (query: {query[:50]}...)")
        return results

    def _update_access_counts(self, memory_ids: List[str]) -> None:
        """
        更新记忆的访问计数
        
        Args:
            memory_ids: 记忆 ID 列表
        """
        if not memory_ids:
            return
        
        try:
            if self.vector_db_type == "qdrant":
                # 获取当前访问计数
                for memory_id in memory_ids:
                    try:
                        # 获取当前点
                        points = self.client.retrieve(
                            collection_name=self.collection_name,
                            ids=[memory_id],
                            with_payload=True,
                        )
                        if points:
                            point = points[0]
                            current_count = point.payload.get("access_count", 0)
                            new_count = current_count + 1
                            
                            # 更新访问计数
                            self.client.set_payload(
                                collection_name=self.collection_name,
                                payload={"access_count": new_count},
                                points=[memory_id],
                            )
                    except Exception as e:
                        logger.warning(f"更新访问计数失败 {memory_id}: {e}")
            elif self.vector_db_type == "chroma":
                # Chroma 的更新方式
                for memory_id in memory_ids:
                    try:
                        # Chroma 需要在 metadata 中更新
                        results = self.collection.get(ids=[memory_id])
                        if results["ids"]:
                            current_metadata = results["metadatas"][0] if results["metadatas"] else {}
                            current_count = current_metadata.get("access_count", 0)
                            new_count = current_count + 1
                            
                            # 更新 metadata
                            self.collection.update(
                                ids=[memory_id],
                                metadatas=[{**current_metadata, "access_count": new_count}],
                            )
                    except Exception as e:
                        logger.warning(f"更新访问计数失败 {memory_id}: {e}")
        except Exception as e:
            logger.warning(f"批量更新访问计数失败: {e}")

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
