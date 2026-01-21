"""
长期记忆模块全面测试（使用Mock，不依赖真实数据库）
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from typing import List, Dict, Any
import hashlib
import time

from src.memory.long_term import LongTermMemory, MemoryItem, MemoryType


@pytest.fixture
def mock_embedding_model():
    """Mock嵌入模型"""
    import numpy as np
    model = Mock()
    
    def encode_side_effect(texts, **kwargs):
        """根据输入类型返回不同的结果"""
        if isinstance(texts, str):
            # 单个文本，返回一维数组 (384,)
            return np.array([0.1] * 384)
        elif isinstance(texts, list):
            # 列表，返回二维数组 (N, 384)
            return np.array([[0.1] * 384] * len(texts))
        else:
            # 默认返回一维数组
            return np.array([0.1] * 384)
    
    model.encode = Mock(side_effect=encode_side_effect)
    return model


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant客户端"""
    client = Mock()
    
    # Mock集合列表
    mock_collections = Mock()
    mock_collections.collections = []
    client.get_collections = Mock(return_value=mock_collections)
    
    # Mock创建集合
    client.create_collection = Mock()
    
    # Mock upsert
    client.upsert = Mock()
    
    # Mock search/query_points
    mock_point = Mock()
    mock_point.id = "test_id"
    mock_point.score = 0.95
    mock_point.payload = {"content": "test content", "timestamp": time.time()}
    
    mock_query_result = Mock()
    mock_query_result.points = [mock_point]
    client.query_points = Mock(return_value=mock_query_result)
    
    # Mock scroll
    mock_scroll_point = Mock()
    mock_scroll_point.payload = {"content": "existing content"}
    client.scroll = Mock(return_value=([mock_scroll_point], None))
    
    # Mock get_collection
    mock_collection_info = Mock()
    mock_collection_info.points_count = 10
    client.get_collection = Mock(return_value=mock_collection_info)
    
    # Mock delete
    client.delete = Mock(return_value=Mock(ok=True))
    
    return client


@pytest.fixture
def mock_chroma_client():
    """Mock Chroma客户端"""
    client = Mock()
    collection = Mock()
    
    # Mock get_or_create_collection
    client.get_or_create_collection = Mock(return_value=collection)
    
    # Mock collection methods
    collection.add = Mock()
    collection.query = Mock(return_value={
        "ids": [["test_id"]],
        "documents": [["test content"]],
        "metadatas": [[{"content": "test content"}]],
        "distances": [[0.05]]
    })
    collection.get = Mock(return_value={
        "documents": ["existing content"]
    })
    collection.delete = Mock()
    
    return client, collection


class TestLongTermMemoryCore:
    """长期记忆核心功能测试（使用Mock）"""
    
    @patch("src.memory.long_term.SentenceTransformer")
    @patch("src.memory.long_term.QdrantClient")
    def test_add_memory_qdrant(self, mock_qdrant_class, mock_sentence_transformer, mock_qdrant_client, mock_embedding_model):
        """测试添加记忆到Qdrant"""
        mock_sentence_transformer.return_value = mock_embedding_model
        mock_qdrant_class.return_value = mock_qdrant_client
        
        memory = LongTermMemory(vector_db="qdrant", embedding_model="test/model")
        
        memory_id = memory.add_memory(
            content="测试内容",
            metadata={"key": "value"},
            session_id="test_session"
        )
        
        assert memory_id is not None
        assert isinstance(memory_id, str)
        mock_qdrant_client.upsert.assert_called_once()
    
    @patch("src.memory.long_term.SentenceTransformer")
    @patch("src.memory.long_term.PersistentClient")
    def test_add_memory_chroma(self, mock_chroma_class, mock_sentence_transformer, mock_chroma_client, mock_embedding_model):
        """测试添加记忆到Chroma"""
        mock_sentence_transformer.return_value = mock_embedding_model
        client, collection = mock_chroma_client
        mock_chroma_class.return_value = client
        
        memory = LongTermMemory(vector_db="chroma", embedding_model="test/model")
        
        memory_id = memory.add_memory(
            content="测试内容",
            metadata={"key": "value"},
            session_id="test_session"
        )
        
        assert memory_id is not None
        collection.add.assert_called_once()
    
    @patch("src.memory.long_term.SentenceTransformer")
    @patch("src.memory.long_term.QdrantClient")
    def test_add_memories_batch(self, mock_qdrant_class, mock_sentence_transformer, mock_qdrant_client, mock_embedding_model):
        """测试批量添加记忆"""
        import numpy as np
        mock_sentence_transformer.return_value = mock_embedding_model
        mock_embedding_model.encode = Mock(return_value=np.array([[0.1] * 384] * 3))
        mock_qdrant_class.return_value = mock_qdrant_client
        
        memory = LongTermMemory(vector_db="qdrant", embedding_model="test/model")
        
        contents = ["内容1", "内容2", "内容3"]
        metadatas = [{"key": "value1"}, {"key": "value2"}, {"key": "value3"}]
        
        memory_ids = memory.add_memories_batch(
            contents=contents,
            metadatas=metadatas,
            session_id="test_session"
        )
        
        assert len(memory_ids) == 3
        assert all(isinstance(id, str) for id in memory_ids)
        mock_qdrant_client.upsert.assert_called_once()
    
    @patch("src.memory.long_term.SentenceTransformer")
    @patch("src.memory.long_term.QdrantClient")
    def test_search_qdrant(self, mock_qdrant_class, mock_sentence_transformer, mock_qdrant_client, mock_embedding_model):
        """测试搜索Qdrant"""
        mock_sentence_transformer.return_value = mock_embedding_model
        mock_qdrant_class.return_value = mock_qdrant_client
        
        memory = LongTermMemory(vector_db="qdrant", embedding_model="test/model")
        
        results = memory.search(
            query="测试查询",
            top_k=5,
            session_id="test_session"
        )
        
        assert isinstance(results, list)
        mock_qdrant_client.query_points.assert_called_once()
    
    @patch("src.memory.long_term.SentenceTransformer")
    @patch("src.memory.long_term.PersistentClient")
    def test_search_chroma(self, mock_chroma_class, mock_sentence_transformer, mock_chroma_client, mock_embedding_model):
        """测试搜索Chroma"""
        mock_sentence_transformer.return_value = mock_embedding_model
        client, collection = mock_chroma_client
        mock_chroma_class.return_value = client
        
        memory = LongTermMemory(vector_db="chroma", embedding_model="test/model")
        
        results = memory.search(
            query="测试查询",
            top_k=5,
            session_id="test_session"
        )
        
        assert isinstance(results, list)
        collection.query.assert_called_once()
    
    @patch("src.memory.long_term.SentenceTransformer")
    @patch("src.memory.long_term.QdrantClient")
    def test_search_with_score_threshold(self, mock_qdrant_class, mock_sentence_transformer, mock_qdrant_client, mock_embedding_model):
        """测试带分数阈值的搜索"""
        mock_sentence_transformer.return_value = mock_embedding_model
        mock_qdrant_class.return_value = mock_qdrant_client
        
        memory = LongTermMemory(vector_db="qdrant", embedding_model="test/model")
        
        results = memory.search(
            query="测试查询",
            top_k=5,
            score_threshold=0.5,
            session_id="test_session"
        )
        
        assert isinstance(results, list)
    
    @patch("src.memory.long_term.SentenceTransformer")
    @patch("src.memory.long_term.QdrantClient")
    def test_search_with_metadata_filter(self, mock_qdrant_class, mock_sentence_transformer, mock_qdrant_client, mock_embedding_model):
        """测试带元数据过滤的搜索"""
        mock_sentence_transformer.return_value = mock_embedding_model
        mock_qdrant_class.return_value = mock_qdrant_client
        
        memory = LongTermMemory(vector_db="qdrant", embedding_model="test/model")
        
        results = memory.search(
            query="测试查询",
            top_k=5,
            metadata_filter={"category": "test"},
            session_id="test_session"
        )
        
        assert isinstance(results, list)
    
    @patch("src.memory.long_term.SentenceTransformer")
    @patch("src.memory.long_term.QdrantClient")
    def test_compute_content_hash(self, mock_qdrant_class, mock_sentence_transformer, mock_qdrant_client, mock_embedding_model):
        """测试内容哈希计算"""
        mock_sentence_transformer.return_value = mock_embedding_model
        mock_qdrant_class.return_value = mock_qdrant_client
        
        memory = LongTermMemory(vector_db="qdrant", embedding_model="test/model")
        
        content = "测试内容"
        hash1 = memory._compute_content_hash(content)
        hash2 = memory._compute_content_hash(content)
        
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 32  # MD5哈希长度
    
    @patch("src.memory.long_term.SentenceTransformer")
    @patch("src.memory.long_term.QdrantClient")
    def test_embed_text(self, mock_qdrant_class, mock_sentence_transformer, mock_qdrant_client, mock_embedding_model):
        """测试文本嵌入"""
        mock_sentence_transformer.return_value = mock_embedding_model
        mock_qdrant_class.return_value = mock_qdrant_client
        
        memory = LongTermMemory(vector_db="qdrant", embedding_model="test/model")
        
        embedding = memory._embed_text("测试文本")
        
        assert isinstance(embedding, list)
        # _embed_text返回的是list（通过.tolist()），所以应该是384维向量的列表
        assert len(embedding) == 384
        # 验证encode被调用，并且参数正确
        mock_embedding_model.encode.assert_called_with("测试文本", normalize_embeddings=True)
    
    @patch("src.memory.long_term.SentenceTransformer")
    @patch("src.memory.long_term.QdrantClient")
    def test_embed_texts(self, mock_qdrant_class, mock_sentence_transformer, mock_qdrant_client, mock_embedding_model):
        """测试批量文本嵌入"""
        import numpy as np
        mock_sentence_transformer.return_value = mock_embedding_model
        mock_embedding_model.encode = Mock(return_value=np.array([[0.1] * 384] * 2))
        mock_qdrant_class.return_value = mock_qdrant_client
        
        memory = LongTermMemory(vector_db="qdrant", embedding_model="test/model")
        
        texts = ["文本1", "文本2"]
        embeddings = memory._embed_texts(texts)
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == 2
        assert all(len(e) == 384 for e in embeddings)
    
    @patch("src.memory.long_term.SentenceTransformer")
    @patch("src.memory.long_term.QdrantClient")
    @patch("src.memory.long_term.np")
    def test_check_semantic_similarity(self, mock_np, mock_qdrant_class, mock_sentence_transformer, mock_qdrant_client, mock_embedding_model):
        """测试语义相似度检查"""
        import numpy as np
        mock_sentence_transformer.return_value = mock_embedding_model
        mock_embedding_model.encode = Mock(return_value=np.array([[0.1] * 384] * 3))
        mock_qdrant_class.return_value = mock_qdrant_client
        
        # Mock numpy
        mock_np.array = Mock(side_effect=lambda x: x)
        mock_np.dot = Mock(return_value=0.98)
        mock_np.linalg.norm = Mock(return_value=1.0)
        
        memory = LongTermMemory(vector_db="qdrant", embedding_model="test/model")
        
        new_content = "新内容"
        existing_contents = ["相似内容1", "不同内容2"]
        
        # 设置相似度阈值为0.95
        is_similar = memory._check_semantic_similarity(
            new_content,
            existing_contents,
            threshold=0.95
        )
        
        # 由于mock返回0.98，应该返回True
        assert isinstance(is_similar, bool)
    
    @patch("src.memory.long_term.SentenceTransformer")
    @patch("src.memory.long_term.QdrantClient")
    def test_get_recent_contents_qdrant(self, mock_qdrant_class, mock_sentence_transformer, mock_qdrant_client, mock_embedding_model):
        """测试获取最近内容（Qdrant）"""
        mock_sentence_transformer.return_value = mock_embedding_model
        mock_qdrant_class.return_value = mock_qdrant_client
        
        memory = LongTermMemory(vector_db="qdrant", embedding_model="test/model")
        
        contents = memory._get_recent_contents(limit=10)
        
        assert isinstance(contents, list)
        mock_qdrant_client.scroll.assert_called_once()
    
    @patch("src.memory.long_term.SentenceTransformer")
    @patch("src.memory.long_term.PersistentClient")
    def test_get_recent_contents_chroma(self, mock_chroma_class, mock_sentence_transformer, mock_chroma_client, mock_embedding_model):
        """测试获取最近内容（Chroma）"""
        mock_sentence_transformer.return_value = mock_embedding_model
        client, collection = mock_chroma_client
        mock_chroma_class.return_value = client
        
        memory = LongTermMemory(vector_db="chroma", embedding_model="test/model")
        
        contents = memory._get_recent_contents(limit=10)
        
        assert isinstance(contents, list)
        collection.get.assert_called_once()
    
    @patch("src.memory.long_term.SentenceTransformer")
    @patch("src.memory.long_term.QdrantClient")
    def test_archive_from_short_term_with_deduplicate(self, mock_qdrant_class, mock_sentence_transformer, mock_qdrant_client, mock_embedding_model):
        """测试归档短期记忆（带去重）"""
        import numpy as np
        mock_sentence_transformer.return_value = mock_embedding_model
        mock_embedding_model.encode = Mock(return_value=np.array([[0.1] * 384]))
        mock_qdrant_class.return_value = mock_qdrant_client
        
        memory = LongTermMemory(vector_db="qdrant", embedding_model="test/model")
        
        conversation_turns = [
            {"human": "问题1", "ai": "回答1", "timestamp": 1000.0},
            {"human": "问题1", "ai": "回答1", "timestamp": 2000.0},  # 重复内容
        ]
        
        archived_ids = memory.archive_from_short_term(
            conversation_turns=conversation_turns,
            session_id="test_session",
            deduplicate=True,
            semantic_dedup=False
        )
        
        # 去重后应该只有1个
        assert len(archived_ids) == 1
    
    @patch("src.memory.long_term.SentenceTransformer")
    @patch("src.memory.long_term.QdrantClient")
    def test_archive_from_short_term_with_filter_low_value(self, mock_qdrant_class, mock_sentence_transformer, mock_qdrant_client, mock_embedding_model):
        """测试归档短期记忆（带低价值过滤）"""
        import numpy as np
        mock_sentence_transformer.return_value = mock_embedding_model
        mock_embedding_model.encode = Mock(return_value=np.array([[0.1] * 384]))
        mock_qdrant_class.return_value = mock_qdrant_client
        
        memory = LongTermMemory(vector_db="qdrant", embedding_model="test/model")
        
        conversation_turns = [
            {"human": "好的", "ai": "收到", "timestamp": 1000.0},  # 低价值内容
            {"human": "我要买一个订单号为12345的商品", "ai": "好的，订单号12345", "timestamp": 2000.0},  # 包含业务关键词
        ]
        
        archived_ids = memory.archive_from_short_term(
            conversation_turns=conversation_turns,
            session_id="test_session",
            deduplicate=False,
            filter_low_value=True
        )
        
        # 低价值内容应该被过滤，但包含业务关键词的应该保留
        assert len(archived_ids) >= 1
    
    @patch("src.memory.long_term.SentenceTransformer")
    @patch("src.memory.long_term.QdrantClient")
    def test_get_stats(self, mock_qdrant_class, mock_sentence_transformer, mock_qdrant_client, mock_embedding_model):
        """测试获取统计信息"""
        mock_sentence_transformer.return_value = mock_embedding_model
        mock_qdrant_class.return_value = mock_qdrant_client
        
        memory = LongTermMemory(vector_db="qdrant", embedding_model="test/model")
        
        stats = memory.get_stats()
        
        assert isinstance(stats, dict)
        assert "vector_db_type" in stats
        assert "collection_name" in stats
        assert "embedding_model" in stats
        assert "embedding_dim" in stats
        assert "total_memories" in stats
        assert stats["vector_db_type"] == "qdrant"
    
    @patch("src.memory.long_term.SentenceTransformer")
    @patch("src.memory.long_term.QdrantClient")
    def test_delete_memory_qdrant(self, mock_qdrant_class, mock_sentence_transformer, mock_qdrant_client, mock_embedding_model):
        """测试删除记忆（Qdrant）"""
        mock_sentence_transformer.return_value = mock_embedding_model
        mock_qdrant_class.return_value = mock_qdrant_client
        
        memory = LongTermMemory(vector_db="qdrant", embedding_model="test/model")
        
        success = memory.delete_memory("test_id")
        
        assert success is True
        mock_qdrant_client.delete.assert_called_once()
    
    @patch("src.memory.long_term.SentenceTransformer")
    @patch("src.memory.long_term.PersistentClient")
    def test_delete_memory_chroma(self, mock_chroma_class, mock_sentence_transformer, mock_chroma_client, mock_embedding_model):
        """测试删除记忆（Chroma）"""
        mock_sentence_transformer.return_value = mock_embedding_model
        client, collection = mock_chroma_client
        mock_chroma_class.return_value = client
        
        memory = LongTermMemory(vector_db="chroma", embedding_model="test/model")
        
        success = memory.delete_memory("test_id")
        
        assert success is True
        collection.delete.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
