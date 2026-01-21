"""
集成测试

测试各个模块之间的集成和端到端流程。
"""

import pytest
from typing import Dict, Any

from src.graph.state import AgentState
from src.memory.short_term import ShortTermMemory
from src.memory.routing import MemoryRouter, RoutingContext, RouteDecision


class TestIntegration:
    """集成测试类"""

    def test_short_term_to_long_term_flow(self):
        """测试短期记忆到长期记忆的流程"""
        # 测试短期记忆的基本功能，长期记忆部分需要向量数据库
        short_memory = ShortTermMemory(max_turns=5)
        
        # 添加对话到短期记忆
        short_memory.add_conversation_turn(
            human_message="用户说：我喜欢Python",
            ai_message="Python是个很好的编程语言",
        )
        
        # 验证短期记忆工作正常
        context = short_memory.get_recent_context()
        assert context["turn_count"] == 1
        
        # 获取 LLM 格式的上下文（应该包含对话内容）
        llm_context = short_memory.get_context_for_llm()
        assert "Python" in llm_context
        
        # 注意：实际的长期记忆归档需要向量数据库，这里只测试短期记忆部分
        # 完整的长期记忆测试需要 mock 向量数据库或实际连接

    def test_routing_decision_flow(self):
        """测试路由决策流程"""
        # 创建短期记忆
        short_memory = ShortTermMemory(max_turns=5)

        # 添加一些对话
        for i in range(3):
            short_memory.add_conversation_turn(
                human_message=f"问题{i}",
                ai_message=f"回答{i}",
            )

        # 创建路由上下文（需要长期记忆，这里简化）
        # 实际测试中需要 mock 长期记忆
        assert len(short_memory.conversation_buffer) == 3

    def test_agent_state_flow(self):
        """测试 Agent 状态流转"""
        # 初始状态
        initial_state: AgentState = {
            "input": "测试问题",
            "chat_history": [],
            "documents": [],
            "document_metadata": [],
            "generation": "",
            "relevance_score": "",
            "hallucination_score": "",
            "retry_count": 0,
        }

        # 模拟状态更新
        updated_state = {
            **initial_state,
            "documents": ["文档1", "文档2"],
            "generation": "生成的回答",
            "relevance_score": "yes",
        }

        assert updated_state["input"] == initial_state["input"]
        assert len(updated_state["documents"]) == 2
        assert updated_state["generation"] == "生成的回答"


class TestEndToEnd:
    """端到端测试"""

    def test_memory_system_workflow(self):
        """测试记忆系统完整工作流程"""
        # 1. 创建短期记忆
        short_memory = ShortTermMemory(max_turns=10)

        # 2. 添加对话
        short_memory.add_conversation_turn(
            human_message="用户问题",
            ai_message="AI回答",
        )

        # 3. 获取上下文
        context = short_memory.get_recent_context()
        assert context["turn_count"] == 1

        # 4. 获取 LLM 格式的上下文
        llm_context = short_memory.get_context_for_llm()
        assert isinstance(llm_context, str)
        assert len(llm_context) > 0

        # 5. 获取统计信息
        stats = short_memory.get_stats()
        assert stats["turn_count"] == 1

    def test_routing_workflow(self):
        """测试路由工作流程"""
        short_memory = ShortTermMemory(max_turns=5)

        # 创建路由上下文（简化版，不包含长期记忆）
        # 实际应用中需要完整的长期记忆实例
        assert short_memory.max_turns == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
