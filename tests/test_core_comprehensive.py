"""
核心模块全面测试（使用Mock，不依赖真实数据库）
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

try:
    from src.core import RAGEnhancedAgentMemory
except ImportError:
    RAGEnhancedAgentMemory = None

from src.graph.state import AgentState


@pytest.fixture
def mock_long_term_memory():
    """Mock长期记忆"""
    memory = Mock()
    memory.add_memory = Mock(return_value="memory_id_123")
    memory.search = Mock(return_value=[])
    memory.archive_from_short_term = Mock(return_value=["id1", "id2"])
    memory.get_stats = Mock(return_value={
        "vector_db_type": "qdrant",
        "total_memories": 10
    })
    return memory


@pytest.fixture
def mock_short_term_memory():
    """Mock短期记忆"""
    memory = Mock()
    memory.add_conversation_turn = Mock()
    memory.get_recent_context = Mock(return_value={
        "conversation": [
            {"human": "问题1", "ai": "回答1", "timestamp": 1000.0}
        ]
    })
    memory.get_context_for_llm = Mock(return_value="上下文文本")
    memory.get_stats = Mock(return_value={"turn_count": 2})
    memory.clear = Mock()
    memory.set_session_id = Mock()
    return memory


@pytest.fixture
def mock_router():
    """Mock路由器"""
    router = Mock()
    mock_result = Mock()
    mock_result.decision = Mock()
    mock_result.decision.value = "short_term_only"
    mock_result.short_term_context = {
        "conversation": [{"human": "问题", "ai": "回答"}]
    }
    mock_result.long_term_results = None
    router.route = Mock(return_value=mock_result)
    return router


class TestCoreComprehensive:
    """核心模块全面测试"""
    
    @patch("src.core.LongTermMemory")
    @patch("src.core.ShortTermMemory")
    @patch("src.core.MemoryRouter")
    def test_initialization_with_all_params(self, mock_router_class, mock_short_term_class, mock_long_term_class):
        """测试使用所有参数初始化"""
        if RAGEnhancedAgentMemory is None:
            pytest.skip("需要 langchain-core 依赖")
        
        mock_router_class.return_value = Mock()
        mock_short_term_class.return_value = Mock()
        mock_long_term_class.return_value = Mock()
        
        memory = RAGEnhancedAgentMemory(
            vector_db="qdrant",
            embedding_model="test/model",
            rerank_model="test/rerank",
            short_term_threshold=5,
            long_term_trigger=0.8,
            use_hybrid_retrieval=True,
            use_rerank=True,
            session_id="test_session"
        )
        
        assert memory.session_id == "test_session"
        assert memory.use_hybrid_retrieval is True
        assert memory.use_rerank is True
    
    @patch("src.core.LongTermMemory")
    @patch("src.core.ShortTermMemory")
    @patch("src.core.MemoryRouter")
    def test_save_context_normal(self, mock_router_class, mock_short_term_class, mock_long_term_class, mock_short_term_memory):
        """测试正常保存上下文"""
        if RAGEnhancedAgentMemory is None:
            pytest.skip("需要 langchain-core 依赖")
        
        mock_router_class.return_value = Mock()
        mock_short_term_class.return_value = mock_short_term_memory
        mock_long_term_class.return_value = Mock()
        
        memory = RAGEnhancedAgentMemory(session_id="test_session")
        
        memory.save_context(
            inputs={"input": "用户问题"},
            outputs={"generation": "AI回答"}
        )
        
        mock_short_term_memory.add_conversation_turn.assert_called_once()
    
    @patch("src.core.LongTermMemory")
    @patch("src.core.ShortTermMemory")
    @patch("src.core.MemoryRouter")
    @patch("src.core.get_settings")
    def test_save_context_with_low_value_filter(self, mock_get_settings, mock_router_class, mock_short_term_class, mock_long_term_class, mock_short_term_memory):
        """测试保存上下文时过滤低价值信息"""
        if RAGEnhancedAgentMemory is None:
            pytest.skip("需要 langchain-core 依赖")
        
        # Mock设置启用低价值过滤
        mock_settings = Mock()
        mock_settings.enable_low_value_filter = True
        mock_get_settings.return_value = mock_settings
        
        mock_router_class.return_value = Mock()
        mock_short_term_class.return_value = mock_short_term_memory
        mock_long_term_class.return_value = Mock()
        
        memory = RAGEnhancedAgentMemory(session_id="test_session")
        
        # 低价值对话（两个都是低价值）
        memory.save_context(
            inputs={"input": "好的"},
            outputs={"generation": "收到"}
        )
        
        # 应该被过滤，不保存
        mock_short_term_memory.add_conversation_turn.assert_not_called()
        
        # 包含业务关键词的应该保存
        memory.save_context(
            inputs={"input": "订单号12345"},
            outputs={"generation": "好的，订单号12345"}
        )
        
        mock_short_term_memory.add_conversation_turn.assert_called_once()
    
    @patch("src.core.LongTermMemory")
    @patch("src.core.ShortTermMemory")
    @patch("src.core.MemoryRouter")
    def test_save_context_empty_inputs(self, mock_router_class, mock_short_term_class, mock_long_term_class, mock_short_term_memory):
        """测试保存空输入"""
        if RAGEnhancedAgentMemory is None:
            pytest.skip("需要 langchain-core 依赖")
        
        mock_router_class.return_value = Mock()
        mock_short_term_class.return_value = mock_short_term_memory
        mock_long_term_class.return_value = Mock()
        
        memory = RAGEnhancedAgentMemory(session_id="test_session")
        
        # 空输入或空输出不应该保存
        memory.save_context(
            inputs={"input": ""},
            outputs={"generation": "回答"}
        )
        
        memory.save_context(
            inputs={"input": "问题"},
            outputs={"generation": ""}
        )
        
        # 都不应该保存
        assert mock_short_term_memory.add_conversation_turn.call_count == 0
    
    @patch("src.core.LongTermMemory")
    @patch("src.core.ShortTermMemory")
    @patch("src.core.MemoryRouter")
    def test_retrieve_context_empty_query(self, mock_router_class, mock_short_term_class, mock_long_term_class):
        """测试空查询的上下文检索"""
        if RAGEnhancedAgentMemory is None:
            pytest.skip("需要 langchain-core 依赖")
        
        mock_router_class.return_value = Mock()
        mock_short_term_class.return_value = Mock()
        mock_long_term_class.return_value = Mock()
        
        memory = RAGEnhancedAgentMemory(session_id="test_session")
        
        result = memory.retrieve_context({"input": ""})
        
        assert result["documents"] == []
        assert result["memory_type"] == "none"
    
    @patch("src.core.LongTermMemory")
    @patch("src.core.ShortTermMemory")
    @patch("src.core.MemoryRouter")
    def test_retrieve_context_short_term_only(self, mock_router_class, mock_short_term_class, mock_long_term_class):
        """测试仅使用短期记忆的检索"""
        if RAGEnhancedAgentMemory is None:
            pytest.skip("需要 langchain-core 依赖")
        
        mock_short_term = Mock()
        mock_short_term.get_recent_context = Mock(return_value={
            "conversation": [{"human": "问题", "ai": "回答"}]
        })
        
        mock_router_result = Mock()
        mock_router_result.decision = Mock()
        mock_router_result.decision.value = "short_term_only"
        mock_router_result.short_term_context = {
            "conversation": [{"human": "问题", "ai": "回答"}]
        }
        mock_router_result.long_term_results = None
        
        mock_router = Mock()
        mock_router.route = Mock(return_value=mock_router_result)
        
        mock_router_class.return_value = mock_router
        mock_short_term_class.return_value = mock_short_term
        mock_long_term_class.return_value = Mock()
        
        memory = RAGEnhancedAgentMemory(session_id="test_session")
        
        result = memory.retrieve_context({"input": "测试查询"})
        
        assert len(result["documents"]) > 0
        assert result["memory_type"] == "short_term_only"
    
    @patch("src.core.LongTermMemory")
    @patch("src.core.ShortTermMemory")
    @patch("src.core.MemoryRouter")
    def test_retrieve_context_long_term_only(self, mock_router_class, mock_short_term_class, mock_long_term_class):
        """测试仅使用长期记忆的检索"""
        if RAGEnhancedAgentMemory is None:
            pytest.skip("需要 langchain-core 依赖")
        
        from src.memory.long_term import MemoryItem, MemoryType
        
        mock_memory_item = MemoryItem(
            id="test_id",
            content="长期记忆内容",
            relevance_score=0.9,
            memory_type=MemoryType.LONG_TERM
        )
        
        mock_router_result = Mock()
        mock_router_result.decision = Mock()
        mock_router_result.decision.value = "long_term_only"
        mock_router_result.short_term_context = None
        mock_router_result.long_term_results = [mock_memory_item]
        
        mock_router = Mock()
        mock_router.route = Mock(return_value=mock_router_result)
        
        mock_router_class.return_value = mock_router
        mock_short_term_class.return_value = Mock()
        mock_long_term_class.return_value = Mock()
        
        memory = RAGEnhancedAgentMemory(session_id="test_session", use_hybrid_retrieval=False)
        
        result = memory.retrieve_context({"input": "测试查询"})
        
        assert len(result["documents"]) > 0
        assert result["memory_type"] == "long_term_only"
        assert "长期记忆内容" in result["documents"]
    
    @patch("src.core.LongTermMemory")
    @patch("src.core.ShortTermMemory")
    @patch("src.core.MemoryRouter")
    @patch("src.core.HybridRetriever")
    def test_retrieve_context_hybrid(self, mock_hybrid_class, mock_router_class, mock_short_term_class, mock_long_term_class):
        """测试混合检索"""
        if RAGEnhancedAgentMemory is None:
            pytest.skip("需要 langchain-core 依赖")
        
        from src.retrieval.hybrid import HybridRetrievalResult
        
        mock_hybrid_result = HybridRetrievalResult(
            content="混合检索内容",
            vector_score=0.8,
            keyword_score=0.7,
            combined_score=0.75,
            source="hybrid",
            metadata={}
        )
        
        mock_hybrid_retriever = Mock()
        mock_hybrid_retriever.retrieve = Mock(return_value=[mock_hybrid_result])
        
        mock_router_result = Mock()
        mock_router_result.decision = Mock()
        mock_router_result.decision.value = "hybrid"
        mock_router_result.short_term_context = None
        mock_router_result.long_term_results = None
        
        mock_router = Mock()
        mock_router.route = Mock(return_value=mock_router_result)
        
        mock_router_class.return_value = mock_router
        mock_short_term_class.return_value = Mock()
        mock_long_term_class.return_value = Mock()
        mock_hybrid_class.return_value = mock_hybrid_retriever
        
        memory = RAGEnhancedAgentMemory(session_id="test_session", use_hybrid_retrieval=True)
        
        result = memory.retrieve_context({"input": "测试查询"})
        
        assert len(result["documents"]) > 0
        assert result["memory_type"] == "hybrid"
    
    @patch("src.core.LongTermMemory")
    @patch("src.core.ShortTermMemory")
    @patch("src.core.MemoryRouter")
    def test_add_memory(self, mock_router_class, mock_short_term_class, mock_long_term_class, mock_long_term_memory):
        """测试添加记忆"""
        if RAGEnhancedAgentMemory is None:
            pytest.skip("需要 langchain-core 依赖")
        
        mock_router_class.return_value = Mock()
        mock_short_term_class.return_value = Mock()
        mock_long_term_class.return_value = mock_long_term_memory
        
        memory = RAGEnhancedAgentMemory(session_id="test_session")
        
        memory_id = memory.add_memory(
            content="测试内容",
            metadata={"key": "value"}
        )
        
        assert memory_id == "memory_id_123"
        mock_long_term_memory.add_memory.assert_called_once()
    
    @patch("src.core.LongTermMemory")
    @patch("src.core.ShortTermMemory")
    @patch("src.core.MemoryRouter")
    def test_search_without_hybrid(self, mock_router_class, mock_short_term_class, mock_long_term_class, mock_long_term_memory):
        """测试不使用混合检索的搜索"""
        if RAGEnhancedAgentMemory is None:
            pytest.skip("需要 langchain-core 依赖")
        
        from src.memory.long_term import MemoryItem, MemoryType
        
        mock_memory_item = MemoryItem(
            id="test_id",
            content="搜索结果",
            relevance_score=0.9,
            memory_type=MemoryType.LONG_TERM,
            metadata={"key": "value"}
        )
        
        mock_long_term_memory.search = Mock(return_value=[mock_memory_item])
        
        mock_router_class.return_value = Mock()
        mock_short_term_class.return_value = Mock()
        mock_long_term_class.return_value = mock_long_term_memory
        
        memory = RAGEnhancedAgentMemory(session_id="test_session", use_hybrid_retrieval=False)
        
        results = memory.search(query="测试查询", top_k=5, use_hybrid=False)
        
        assert len(results) > 0
        assert results[0]["content"] == "搜索结果"
        assert results[0]["score"] == 0.9
    
    @patch("src.core.LongTermMemory")
    @patch("src.core.ShortTermMemory")
    @patch("src.core.MemoryRouter")
    @patch("src.core.HybridRetriever")
    def test_search_with_hybrid(self, mock_hybrid_class, mock_router_class, mock_short_term_class, mock_long_term_class):
        """测试使用混合检索的搜索"""
        if RAGEnhancedAgentMemory is None:
            pytest.skip("需要 langchain-core 依赖")
        
        from src.retrieval.hybrid import HybridRetrievalResult
        
        mock_hybrid_result = HybridRetrievalResult(
            content="混合搜索结果",
            vector_score=0.8,
            keyword_score=0.7,
            combined_score=0.75,
            source="hybrid",
            metadata={"key": "value"}
        )
        
        mock_hybrid_retriever = Mock()
        mock_hybrid_retriever.retrieve = Mock(return_value=[mock_hybrid_result])
        
        mock_router_class.return_value = Mock()
        mock_short_term_class.return_value = Mock()
        mock_long_term_class.return_value = Mock()
        mock_hybrid_class.return_value = mock_hybrid_retriever
        
        memory = RAGEnhancedAgentMemory(session_id="test_session", use_hybrid_retrieval=True)
        
        results = memory.search(query="测试查询", top_k=5, use_hybrid=True)
        
        assert len(results) > 0
        assert results[0]["content"] == "混合搜索结果"
        assert results[0]["source"] == "hybrid"
        assert results[0]["score"] == 0.75
    
    @patch("src.core.LongTermMemory")
    @patch("src.core.ShortTermMemory")
    @patch("src.core.MemoryRouter")
    def test_search_default_behavior(self, mock_router_class, mock_short_term_class, mock_long_term_class, mock_long_term_memory):
        """测试搜索的默认行为"""
        if RAGEnhancedAgentMemory is None:
            pytest.skip("需要 langchain-core 依赖")
        
        from src.memory.long_term import MemoryItem, MemoryType
        
        mock_memory_item = MemoryItem(
            id="test_id",
            content="搜索结果",
            relevance_score=0.9,
            memory_type=MemoryType.LONG_TERM
        )
        
        mock_long_term_memory.search = Mock(return_value=[mock_memory_item])
        
        mock_router_class.return_value = Mock()
        mock_short_term_class.return_value = Mock()
        mock_long_term_class.return_value = mock_long_term_memory
        
        # 创建时不启用混合检索
        memory = RAGEnhancedAgentMemory(session_id="test_session", use_hybrid_retrieval=False)
        
        # 不指定use_hybrid参数，应该使用初始化时的设置
        results = memory.search(query="测试查询", top_k=5)
        
        assert len(results) > 0
        # 应该使用纯向量检索
        mock_long_term_memory.search.assert_called_once()
    
    @patch("src.core.LongTermMemory")
    @patch("src.core.ShortTermMemory")
    @patch("src.core.MemoryRouter")
    def test_get_stats(self, mock_router_class, mock_short_term_class, mock_long_term_class, mock_long_term_memory, mock_short_term_memory):
        """测试获取统计信息"""
        if RAGEnhancedAgentMemory is None:
            pytest.skip("需要 langchain-core 依赖")
        
        mock_router_class.return_value = Mock()
        mock_short_term_class.return_value = mock_short_term_memory
        mock_long_term_class.return_value = mock_long_term_memory
        
        memory = RAGEnhancedAgentMemory(session_id="test_session")
        
        stats = memory.get_stats()
        
        assert isinstance(stats, dict)
        assert "session_id" in stats
        assert "vector_db" in stats
        assert "short_term" in stats
        assert "long_term" in stats
        assert stats["session_id"] == "test_session"
    
    @patch("src.core.LongTermMemory")
    @patch("src.core.ShortTermMemory")
    @patch("src.core.MemoryRouter")
    def test_clear_short_term_memory(self, mock_router_class, mock_short_term_class, mock_long_term_class, mock_short_term_memory):
        """测试清空短期记忆"""
        if RAGEnhancedAgentMemory is None:
            pytest.skip("需要 langchain-core 依赖")
        
        mock_router_class.return_value = Mock()
        mock_short_term_class.return_value = mock_short_term_memory
        mock_long_term_class.return_value = Mock()
        
        memory = RAGEnhancedAgentMemory(session_id="test_session")
        
        memory.clear_short_term_memory()
        
        mock_short_term_memory.clear.assert_called_once()
    
    @patch("src.core.LongTermMemory")
    @patch("src.core.ShortTermMemory")
    @patch("src.core.MemoryRouter")
    @patch("src.core.get_settings")
    def test_archive_short_term_to_long_term(self, mock_get_settings, mock_router_class, mock_short_term_class, mock_long_term_class, mock_long_term_memory, mock_short_term_memory):
        """测试归档短期记忆到长期记忆"""
        if RAGEnhancedAgentMemory is None:
            pytest.skip("需要 langchain-core 依赖")
        
        mock_settings = Mock()
        mock_settings.enable_low_value_filter = False
        mock_get_settings.return_value = mock_settings
        
        mock_router_class.return_value = Mock()
        mock_short_term_class.return_value = mock_short_term_memory
        mock_long_term_class.return_value = mock_long_term_memory
        
        memory = RAGEnhancedAgentMemory(session_id="test_session")
        
        archived_ids = memory.archive_short_term_to_long_term()
        
        assert isinstance(archived_ids, list)
        assert len(archived_ids) == 2
        mock_long_term_memory.archive_from_short_term.assert_called_once()
    
    @patch("src.core.LongTermMemory")
    @patch("src.core.ShortTermMemory")
    @patch("src.core.MemoryRouter")
    def test_archive_empty_short_term(self, mock_router_class, mock_short_term_class, mock_long_term_class, mock_long_term_memory, mock_short_term_memory):
        """测试归档空的短期记忆"""
        if RAGEnhancedAgentMemory is None:
            pytest.skip("需要 langchain-core 依赖")
        
        # Mock空的短期记忆
        mock_short_term_memory.get_recent_context = Mock(return_value={
            "conversation": []
        })
        
        mock_router_class.return_value = Mock()
        mock_short_term_class.return_value = mock_short_term_memory
        mock_long_term_class.return_value = mock_long_term_memory
        
        memory = RAGEnhancedAgentMemory(session_id="test_session")
        
        archived_ids = memory.archive_short_term_to_long_term()
        
        assert archived_ids == []
        # 不应该调用归档方法
        mock_long_term_memory.archive_from_short_term.assert_not_called()
    
    @patch("src.core.LongTermMemory")
    @patch("src.core.ShortTermMemory")
    @patch("src.core.MemoryRouter")
    def test_get_context_for_llm(self, mock_router_class, mock_short_term_class, mock_long_term_class, mock_short_term_memory):
        """测试获取LLM格式的上下文"""
        if RAGEnhancedAgentMemory is None:
            pytest.skip("需要 langchain-core 依赖")
        
        mock_router_class.return_value = Mock()
        mock_short_term_class.return_value = mock_short_term_memory
        mock_long_term_class.return_value = Mock()
        
        memory = RAGEnhancedAgentMemory(session_id="test_session")
        
        context = memory.get_context_for_llm()
        
        assert context == "上下文文本"
        mock_short_term_memory.get_context_for_llm.assert_called_once()
    
    @patch("src.core.LongTermMemory")
    @patch("src.core.ShortTermMemory")
    @patch("src.core.MemoryRouter")
    def test_get_context_for_llm_with_max_tokens(self, mock_router_class, mock_short_term_class, mock_long_term_class, mock_short_term_memory):
        """测试带最大token限制的LLM上下文"""
        if RAGEnhancedAgentMemory is None:
            pytest.skip("需要 langchain-core 依赖")
        
        mock_router_class.return_value = Mock()
        mock_short_term_class.return_value = mock_short_term_memory
        mock_long_term_class.return_value = Mock()
        
        memory = RAGEnhancedAgentMemory(session_id="test_session")
        
        context = memory.get_context_for_llm(max_tokens=100)
        
        assert context == "上下文文本"
        mock_short_term_memory.get_context_for_llm.assert_called_once_with(max_tokens=100)
    
    @patch("src.core.LongTermMemory")
    @patch("src.core.ShortTermMemory")
    @patch("src.core.MemoryRouter")
    def test_is_low_value_message(self, mock_router_class, mock_short_term_class, mock_long_term_class):
        """测试低价值消息判断"""
        if RAGEnhancedAgentMemory is None:
            pytest.skip("需要 langchain-core 依赖")
        
        mock_router_class.return_value = Mock()
        mock_short_term_class.return_value = Mock()
        mock_long_term_class.return_value = Mock()
        
        memory = RAGEnhancedAgentMemory(session_id="test_session")
        
        # 测试低价值消息
        assert memory._is_low_value_message("好的") is True
        assert memory._is_low_value_message("收到") is True
        assert memory._is_low_value_message("嗯") is True
        
        # 测试包含业务关键词的消息（应该保留）
        assert memory._is_low_value_message("订单号12345") is False
        assert memory._is_low_value_message("我要买一个商品") is False
        assert memory._is_low_value_message("多少钱") is False
        assert memory._is_low_value_message("地址在哪里") is False
        
        # 测试正常消息
        assert memory._is_low_value_message("请问如何使用这个功能？") is False
        assert memory._is_low_value_message("我需要帮助") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
