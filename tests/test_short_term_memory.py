"""
测试短期记忆管理模块
"""

import pytest
import time
from langchain_core.messages import HumanMessage, AIMessage

from src.memory.short_term import ShortTermMemory, ConversationTurn, KeyEntity


def test_short_term_memory_initialization():
    """测试短期记忆初始化"""
    memory = ShortTermMemory(max_turns=10)
    
    assert memory.max_turns == 10
    assert len(memory.conversation_buffer) == 0
    assert len(memory.key_entities) == 0
    assert memory.session_id is None


def test_add_conversation_turn():
    """测试添加对话轮次"""
    memory = ShortTermMemory(max_turns=5)
    
    memory.add_conversation_turn(
        human_message="你好",
        ai_message="你好！有什么可以帮助你的吗？"
    )
    
    assert len(memory.conversation_buffer) == 1
    turn = memory.conversation_buffer[0]
    assert turn.human_message == "你好"
    assert turn.ai_message == "你好！有什么可以帮助你的吗？"


def test_sliding_window():
    """测试滑动窗口机制"""
    memory = ShortTermMemory(max_turns=3)
    
    # 添加超过最大轮数的对话
    for i in range(5):
        memory.add_conversation_turn(
            human_message=f"问题{i}",
            ai_message=f"回答{i}"
        )
    
    # 应该只保留最近3轮
    assert len(memory.conversation_buffer) == 3
    assert memory.conversation_buffer[0].human_message == "问题2"
    assert memory.conversation_buffer[-1].human_message == "问题4"


def test_add_messages():
    """测试从 LangChain 消息添加对话"""
    memory = ShortTermMemory(max_turns=5)
    
    messages = [
        HumanMessage(content="用户问题"),
        AIMessage(content="AI回答"),
    ]
    
    memory.add_messages(messages)
    
    assert len(memory.conversation_buffer) == 1
    turn = memory.conversation_buffer[0]
    assert turn.human_message == "用户问题"
    assert turn.ai_message == "AI回答"


def test_get_recent_context():
    """测试获取最近上下文"""
    memory = ShortTermMemory(max_turns=10)
    
    # 添加几轮对话
    for i in range(3):
        memory.add_conversation_turn(
            human_message=f"问题{i}",
            ai_message=f"回答{i}"
        )
    
    context = memory.get_recent_context(n_turns=2)
    
    assert context["turn_count"] == 2
    assert context["total_turns"] == 3
    assert len(context["conversation"]) == 2
    assert context["conversation"][0]["human"] == "问题1"
    assert context["conversation"][-1]["human"] == "问题2"


def test_entity_extraction():
    """测试实体提取"""
    memory = ShortTermMemory(max_turns=5)
    
    memory.add_conversation_turn(
        human_message='我想了解关于"Python"编程语言',
        ai_message="Python是一种高级编程语言"
    )
    
    # 检查是否提取到了实体
    turn = memory.conversation_buffer[0]
    assert len(turn.entities) > 0


def test_entity_cache():
    """测试实体缓存"""
    memory = ShortTermMemory(max_turns=5)
    
    # 添加包含实体的对话
    memory.add_conversation_turn(
        human_message='用户提到了"Python"',
        ai_message="Python是一种编程语言"
    )
    
    # 再次提到相同实体
    memory.add_conversation_turn(
        human_message='"Python"有哪些特性？',
        ai_message="Python有简洁的语法"
    )
    
    # 检查实体缓存
    assert len(memory.key_entities) > 0


def test_compression():
    """测试压缩功能"""
    memory = ShortTermMemory(max_turns=10, compression_threshold=5)
    
    # 添加超过压缩阈值的对话
    for i in range(10):
        memory.add_conversation_turn(
            human_message=f"问题{i}",
            ai_message=f"回答{i}"
        )
    
    # 手动触发压缩
    memory.compress(method="summary")
    
    # 压缩后应该保留一定数量的对话
    assert len(memory.conversation_buffer) <= memory.max_turns


def test_get_context_for_llm():
    """测试获取 LLM 格式的上下文"""
    memory = ShortTermMemory(max_turns=5)
    
    memory.add_conversation_turn(
        human_message="问题1",
        ai_message="回答1"
    )
    
    context_text = memory.get_context_for_llm()
    
    assert isinstance(context_text, str)
    assert "问题1" in context_text
    assert "回答1" in context_text


def test_clear():
    """测试清空记忆"""
    memory = ShortTermMemory(max_turns=5)
    
    # 添加一些对话
    for i in range(3):
        memory.add_conversation_turn(
            human_message=f"问题{i}",
            ai_message=f"回答{i}"
        )
    
    # 清空
    memory.clear()
    
    assert len(memory.conversation_buffer) == 0
    assert len(memory.key_entities) == 0


def test_get_stats():
    """测试获取统计信息"""
    memory = ShortTermMemory(max_turns=10)
    
    memory.add_conversation_turn(
        human_message="问题",
        ai_message="回答"
    )
    
    stats = memory.get_stats()
    
    assert stats["turn_count"] == 1
    assert stats["max_turns"] == 10
    assert "entity_count" in stats
    assert "created_at" in stats


def test_session_id():
    """测试会话 ID 设置"""
    memory = ShortTermMemory(max_turns=5)
    
    memory.set_session_id("test_session_123")
    
    assert memory.session_id == "test_session_123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
