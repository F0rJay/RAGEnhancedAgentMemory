#!/usr/bin/env python3
"""
LangGraph 集成示例

展示如何将 RAGEnhancedAgentMemory 集成到 LangGraph 中。
"""

import sys
from pathlib import Path
from typing import Dict, Any

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core import RAGEnhancedAgentMemory
from src.graph.state import AgentState

try:
    from langgraph.graph import StateGraph
    from langchain_core.messages import HumanMessage, AIMessage
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("警告: LangGraph 未安装，无法运行此示例")
    print("请运行: pip install langgraph langchain-core")


def create_simple_agent(memory: RAGEnhancedAgentMemory):
    """创建简单的 Agent"""
    if not LANGGRAPH_AVAILABLE:
        return None

    def retrieve_node(state: AgentState) -> Dict[str, Any]:
        """检索节点"""
        print(f"   [检索节点] 查询: {state['input'][:50]}...")
        result = memory.retrieve_context(state)
        print(f"   [检索节点] 检索到 {len(result.get('documents', []))} 条文档")
        return result

    def generate_node(state: AgentState) -> Dict[str, Any]:
        """生成节点（简化版）"""
        query = state["input"]
        documents = state.get("documents", [])
        
        # 构建上下文（简化版）
        context = "\n".join(documents[:3]) if documents else "无相关上下文"
        
        # 模拟 LLM 生成（实际应用中调用真实的 LLM）
        generation = f"基于上下文回答: {query}\n相关上下文: {context[:100]}..."
        
        print(f"   [生成节点] 生成回答")
        
        return {
            "generation": generation,
            "chat_history": [
                HumanMessage(content=query),
                AIMessage(content=generation),
            ],
        }

    def validate_node(state: AgentState) -> Dict[str, Any]:
        """验证节点（简化版）"""
        documents = state.get("documents", [])
        generation = state.get("generation", "")
        
        relevance_score = "yes" if documents and generation else "no"
        hallucination_score = "grounded" if documents else "not_grounded"
        
        print(f"   [验证节点] 相关性: {relevance_score}, 幻觉: {hallucination_score}")
        
        return {
            "relevance_score": relevance_score,
            "hallucination_score": hallucination_score,
        }

    # 构建图
    graph = StateGraph(AgentState)
    
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("validate", validate_node)
    
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "validate")
    
    return graph.compile(checkpointer=memory.get_checkpointer())


def main():
    """主函数"""
    if not LANGGRAPH_AVAILABLE:
        print("请先安装 LangGraph: pip install langgraph langchain-core")
        return

    print("=== LangGraph 集成示例 ===\n")

    # 初始化记忆系统
    print("1. 初始化记忆系统...")
    memory = RAGEnhancedAgentMemory(
        vector_db="qdrant",
        session_id="langgraph_demo",
    )
    print("   ✓ 初始化完成\n")

    # 添加一些记忆
    print("2. 添加初始记忆...")
    memory.add_memory("用户喜欢Python编程")
    memory.add_memory("用户使用Django框架")
    print("   ✓ 记忆已添加\n")

    # 创建 Agent
    print("3. 创建 LangGraph Agent...")
    app = create_simple_agent(memory)
    print("   ✓ Agent 创建完成\n")

    # 运行 Agent
    print("4. 运行 Agent...")
    initial_state: AgentState = {
        "input": "用户的技术栈是什么？",
        "chat_history": [],
        "documents": [],
        "document_metadata": [],
        "generation": "",
        "relevance_score": "",
        "hallucination_score": "",
        "retry_count": 0,
    }

    # LangGraph checkpointer 需要 config 参数，包含 thread_id
    config = {"configurable": {"thread_id": memory.session_id}}
    result = app.invoke(initial_state, config=config)
    
    print("\n=== Agent 运行结果 ===")
    print(f"输入: {result['input']}")
    print(f"生成: {result['generation'][:100]}...")
    print(f"相关性评分: {result['relevance_score']}")
    print(f"幻觉评分: {result['hallucination_score']}")

    # 保存对话
    print("\n5. 保存对话到记忆...")
    memory.save_context(
        inputs={"input": initial_state["input"]},
        outputs={"generation": result["generation"]},
    )
    print("   ✓ 对话已保存\n")

    print("=== 示例完成 ===")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"错误: 缺少依赖。请运行: pip install -r requirements.txt")
        print(f"详细错误: {e}")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
