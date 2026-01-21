"""
完整的长期记忆功能测试

使用真实的向量数据库（Qdrant 本地嵌入式模式）进行完整测试
"""

"""
完整的长期记忆功能测试

使用真实的向量数据库（Qdrant 本地嵌入式模式）进行完整测试
"""

import pytest
import os

# 设置离线模式以使用本地模型
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from src.memory.long_term import LongTermMemory, MemoryItem, MemoryType


@pytest.fixture(scope="module")
def long_term_memory():
    """创建长期记忆实例（使用真实的Qdrant本地嵌入式模式）"""
    try:
        # 确保使用离线模式
        import os
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        
        # 使用测试专用的集合名称，避免与生产数据冲突
        memory = LongTermMemory(
            vector_db="qdrant",
            embedding_model="BAAI/bge-large-zh-v1.5",
            collection_name="test_memory_full",
        )
        
        yield memory
    except Exception as e:
        pytest.skip(f"无法初始化长期记忆（可能需要安装依赖或模型）: {e}")


class TestLongTermMemoryFull:
    """完整的长期记忆测试类"""

    def test_add_memory(self, long_term_memory):
        """测试添加单个记忆"""
        memory_id = long_term_memory.add_memory(
            content="用户喜欢Python编程，特别是数据科学领域",
            metadata={"category": "preference", "topic": "programming"},
            session_id="test_session_1",
        )
        
        assert memory_id is not None
        assert isinstance(memory_id, str)
        assert len(memory_id) > 0
        
        # 验证记忆已添加
        stats = long_term_memory.get_stats()
        assert stats["total_memories"] >= 1

    def test_add_multiple_memories(self, long_term_memory):
        """测试添加多个记忆"""
        memories = [
            ("用户最喜欢的编程语言是Python", {"topic": "programming"}),
            ("用户经常使用机器学习库如TensorFlow和PyTorch", {"topic": "ml"}),
            ("用户对自然语言处理很感兴趣", {"topic": "nlp"}),
            ("用户喜欢阅读技术博客和论文", {"topic": "reading"}),
            ("用户最近在学习深度学习", {"topic": "learning"}),
        ]
        
        memory_ids = []
        for content, metadata in memories:
            memory_id = long_term_memory.add_memory(
                content=content,
                metadata=metadata,
                session_id="test_session_1",
            )
            memory_ids.append(memory_id)
        
        assert len(memory_ids) == len(memories)
        assert all(mid is not None for mid in memory_ids)
        
        # 验证所有记忆都已添加
        stats = long_term_memory.get_stats()
        assert stats["total_memories"] >= len(memories)

    def test_search_memories(self, long_term_memory):
        """测试搜索记忆"""
        # 先添加一些记忆
        long_term_memory.add_memory(
            content="Python是用于数据科学的最佳编程语言",
            metadata={"category": "fact"},
            session_id="test_session_1",
        )
        
        # 执行搜索
        results = long_term_memory.search(
            query="编程语言",
            top_k=5,
            session_id="test_session_1",
        )
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        # 验证结果结构
        for result in results:
            assert isinstance(result, MemoryItem)
            assert result.id is not None
            assert result.content is not None
            assert result.relevance_score is not None
            assert result.memory_type == MemoryType.LONG_TERM

    def test_search_with_score_threshold(self, long_term_memory):
        """测试带分数阈值的搜索"""
        results = long_term_memory.search(
            query="机器学习",
            top_k=10,
            score_threshold=0.3,  # 只返回相似度 > 0.3 的结果
            session_id="test_session_1",
        )
        
        assert isinstance(results, list)
        # 验证所有结果的分数都 >= 阈值
        for result in results:
            if result.relevance_score is not None:
                assert result.relevance_score >= 0.3

    def test_search_with_metadata_filter(self, long_term_memory):
        """测试带元数据过滤的搜索"""
        # 先添加带特定元数据的记忆
        long_term_memory.add_memory(
            content="TensorFlow是Google开发的机器学习框架",
            metadata={"topic": "ml", "framework": "tensorflow"},
            session_id="test_session_2",
        )
        
        # 使用元数据过滤搜索
        results = long_term_memory.search(
            query="机器学习框架",
            top_k=5,
            metadata_filter={"topic": "ml"},
            session_id="test_session_2",
        )
        
        assert isinstance(results, list)
        # 验证结果的元数据符合过滤条件
        for result in results:
            if "topic" in result.metadata:
                assert result.metadata["topic"] == "ml"

    def test_search_with_session_filter(self, long_term_memory):
        """测试按会话ID过滤搜索"""
        session_1_content = "这是会话1的内容"
        session_2_content = "这是会话2的内容"
        
        long_term_memory.add_memory(
            content=session_1_content,
            metadata={},
            session_id="session_filter_test_1",
        )
        long_term_memory.add_memory(
            content=session_2_content,
            metadata={},
            session_id="session_filter_test_2",
        )
        
        # 只搜索 session_1 的内容
        results = long_term_memory.search(
            query="会话",
            top_k=10,
            session_id="session_filter_test_1",
        )
        
        # 验证所有结果都属于指定会话
        for result in results:
            assert result.metadata.get("session_id") == "session_filter_test_1"

    def test_add_memories_batch(self, long_term_memory):
        """测试批量添加记忆"""
        contents = [
            "批量记忆1：关于Python的基础知识",
            "批量记忆2：关于机器学习的实践",
            "批量记忆3：关于深度学习的理论",
            "批量记忆4：关于自然语言处理的应用",
            "批量记忆5：关于计算机视觉的进展",
        ]
        
        metadatas = [
            {"batch": "1", "topic": "python"},
            {"batch": "1", "topic": "ml"},
            {"batch": "1", "topic": "dl"},
            {"batch": "1", "topic": "nlp"},
            {"batch": "1", "topic": "cv"},
        ]
        
        memory_ids = long_term_memory.add_memories_batch(
            contents=contents,
            metadatas=metadatas,
            session_id="batch_test_session",
        )
        
        assert len(memory_ids) == len(contents)
        assert all(mid is not None for mid in memory_ids)

    def test_archive_from_short_term(self, long_term_memory):
        """测试从短期记忆归档"""
        conversation_turns = [
            {
                "human": "用户问题1",
                "ai": "AI回答1",
                "timestamp": 1000.0,
            },
            {
                "human": "用户问题2",
                "ai": "AI回答2",
                "timestamp": 2000.0,
            },
            {
                "human": "用户问题3",
                "ai": "AI回答3",
                "timestamp": 3000.0,
            },
        ]
        
        archived_ids = long_term_memory.archive_from_short_term(
            conversation_turns=conversation_turns,
            session_id="archive_test_session",
            deduplicate=True,
        )
        
        assert len(archived_ids) == len(conversation_turns)
        
        # 验证归档的内容可以检索到
        results = long_term_memory.search(
            query="用户问题",
            top_k=10,
            session_id="archive_test_session",
        )
        assert len(results) >= len(conversation_turns)

    def test_deduplication(self, long_term_memory):
        """测试去重功能"""
        content = "这是需要去重的内容"
        
        # 添加相同内容两次
        id1 = long_term_memory.add_memory(
            content=content,
            metadata={},
            session_id="dedup_test",
        )
        
        id2 = long_term_memory.add_memory(
            content=content,
            metadata={},
            session_id="dedup_test",
        )
        
        # 两次应该都成功（因为只是内容相同，ID不同）
        assert id1 != id2
        
        # 但是通过归档去重应该只添加一次
        conversation_turns = [
            {"human": content, "ai": "回答", "timestamp": 1000.0},
            {"human": content, "ai": "回答", "timestamp": 2000.0},  # 相同内容
        ]
        
        archived_ids = long_term_memory.archive_from_short_term(
            conversation_turns=conversation_turns,
            session_id="dedup_test",
            deduplicate=True,
        )
        
        # 去重后应该只有1个
        assert len(archived_ids) == 1

    def test_delete_memory(self, long_term_memory):
        """测试删除记忆"""
        # 先添加一个记忆
        memory_id = long_term_memory.add_memory(
            content="这是要删除的记忆",
            metadata={},
            session_id="delete_test",
        )
        
        # 获取删除前的统计
        stats_before = long_term_memory.get_stats()
        count_before = stats_before["total_memories"]
        
        # 删除记忆
        success = long_term_memory.delete_memory(memory_id)
        assert success is True
        
        # 验证记忆已删除（搜索不应该找到）
        results = long_term_memory.search(
            query="要删除的记忆",
            top_k=10,
            session_id="delete_test",
        )
        # 结果中不应该包含被删除的记忆
        deleted_ids = [r.id for r in results if r.id == memory_id]
        assert len(deleted_ids) == 0

    def test_get_stats(self, long_term_memory):
        """测试获取统计信息"""
        stats = long_term_memory.get_stats()
        
        assert isinstance(stats, dict)
        assert "vector_db_type" in stats
        assert "collection_name" in stats
        assert "embedding_model" in stats
        assert "embedding_dim" in stats
        assert "total_memories" in stats
        
        assert stats["vector_db_type"] == "qdrant"
        assert stats["collection_name"] == "test_memory_full"
        assert stats["embedding_dim"] > 0
        assert isinstance(stats["total_memories"], int)

    def test_search_relevance_ordering(self, long_term_memory):
        """测试搜索结果的相关性排序"""
        # 添加不同相关性的记忆
        memories = [
            ("Python编程语言", {}),
            ("Java编程语言", {}),
            ("C++编程语言", {}),
            ("机器学习算法", {}),
            ("深度学习网络", {}),
        ]
        
        for content, metadata in memories:
            long_term_memory.add_memory(
                content=content,
                metadata=metadata,
                session_id="ordering_test",
            )
        
        # 搜索"Python"，结果应该按相关性排序
        results = long_term_memory.search(
            query="Python",
            top_k=5,
            session_id="ordering_test",
        )
        
        assert len(results) > 0
        
        # 验证分数是降序排列的
        scores = [r.relevance_score for r in results if r.relevance_score is not None]
        if len(scores) > 1:
            assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_complex_search_scenario(self, long_term_memory):
        """测试复杂的搜索场景"""
        # 添加多种类型的记忆
        test_memories = [
            ("用户的姓名是张三，工作是一名软件工程师", {"type": "personal", "field": "name"}),
            ("用户住在北京市海淀区", {"type": "personal", "field": "location"}),
            ("用户最喜欢的编程语言是Python和JavaScript", {"type": "preference", "field": "programming"}),
            ("用户经常使用Docker进行容器化部署", {"type": "skill", "field": "devops"}),
            ("用户对分布式系统设计有深入理解", {"type": "skill", "field": "architecture"}),
        ]
        
        for content, metadata in test_memories:
            long_term_memory.add_memory(
                content=content,
                metadata=metadata,
                session_id="complex_test",
            )
        
        # 测试1: 搜索个人信息
        personal_results = long_term_memory.search(
            query="用户信息",
            top_k=5,
            metadata_filter={"type": "personal"},
            session_id="complex_test",
        )
        assert len(personal_results) > 0
        
        # 测试2: 搜索技能相关
        skill_results = long_term_memory.search(
            query="技术技能",
            top_k=5,
            metadata_filter={"type": "skill"},
            session_id="complex_test",
        )
        assert len(skill_results) > 0
        
        # 测试3: 综合搜索
        all_results = long_term_memory.search(
            query="编程",
            top_k=10,
            session_id="complex_test",
        )
        assert len(all_results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
