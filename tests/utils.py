"""
测试工具函数

提供测试所需的辅助函数和 mock 对象。
"""

from typing import List, Dict, Any, Optional
from unittest.mock import Mock, MagicMock


def create_mock_short_term_memory():
    """创建模拟的短期记忆对象"""
    mock_memory = Mock()
    mock_memory.conversation_buffer = []
    mock_memory.max_turns = 10
    mock_memory.get_recent_context.return_value = {
        "conversation": [],
        "turn_count": 0,
        "total_turns": 0,
    }
    mock_memory.get_stats.return_value = {
        "turn_count": 0,
        "max_turns": 10,
    }
    return mock_memory


def create_mock_long_term_memory():
    """创建模拟的长期记忆对象"""
    mock_memory = Mock()
    mock_memory.search.return_value = []
    mock_memory.add_memory.return_value = "mock_memory_id"
    mock_memory.archive_from_short_term.return_value = []
    mock_memory.get_stats.return_value = {
        "total_memories": 0,
    }
    return mock_memory


def create_mock_agent_state(
    input_text: str = "测试问题",
    documents: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """创建模拟的 Agent 状态"""
    return {
        "input": input_text,
        "chat_history": [],
        "documents": documents or [],
        "document_metadata": [],
        "generation": "",
        "relevance_score": "",
        "hallucination_score": "",
        "retry_count": 0,
    }


def create_test_conversation_turns(count: int = 3) -> List[Dict[str, Any]]:
    """创建测试用的对话轮次"""
    turns = []
    for i in range(count):
        turns.append({
            "human": f"用户问题{i}",
            "ai": f"AI回答{i}",
            "timestamp": 1000.0 + i,
        })
    return turns
