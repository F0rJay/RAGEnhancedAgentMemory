"""
RAG 增强型 Agent 记忆系统使用示例

展示如何使用 RAGEnhancedAgentMemory 构建完整的 Agent 系统。
"""

# 注意：此文件仅作为示例，实际使用时需要安装依赖和启动向量数据库

# from src.core import RAGEnhancedAgentMemory
# from src.graph.state import AgentState
# from langgraph.graph import StateGraph
# from langchain_core.messages import HumanMessage, AIMessage
# 
# 
# def example_basic_usage():
#     """基础使用示例"""
#     # 初始化记忆系统
#     memory = RAGEnhancedAgentMemory(
#         vector_db="qdrant",
#         embedding_model="BAAI/bge-large-en-v1.5",
#         rerank_model="BAAI/bge-reranker-large",
#         session_id="session_001",
#     )
# 
#     # 添加记忆
#     memory_id = memory.add_memory(
#         content="用户喜欢使用 Python 编程语言",
#         metadata={"category": "preference"},
#     )
#     print(f"添加记忆: {memory_id}")
# 
#     # 搜索记忆
#     results = memory.search("用户的技术偏好", top_k=3)
#     print(f"\n搜索结果: {len(results)} 条")
#     for result in results:
#         print(f"  - {result['content'][:50]}... (评分: {result['score']:.2f})")
# 
#     # 获取统计信息
#     stats = memory.get_stats()
#     print(f"\n系统统计: {stats}")
# 
# 
# def example_langgraph_integration():
#     """LangGraph 集成示例"""
#     # 初始化记忆系统
#     memory = RAGEnhancedAgentMemory(
#         vector_db="qdrant",
#         session_id="session_001",
#     )
# 
#     # 定义节点函数
#     def retrieve_node(state: AgentState) -> Dict[str, Any]:
#         """检索节点"""
#         result = memory.retrieve_context(state)
#         return result
# 
#     def generate_node(state: AgentState) -> Dict[str, Any]:
#         """生成节点（简化版）"""
#         query = state["input"]
#         documents = state.get("documents", [])
#         
#         # 构建上下文
#         context = "\n".join(documents[:3])  # 使用前3个文档
#         
#         # 模拟 LLM 生成（实际应用中调用 vLLM 或其他 LLM）
#         generation = f"基于上下文回答: {query}"
#         
#         return {
#             "generation": generation,
#             "chat_history": [
#                 HumanMessage(content=query),
#                 AIMessage(content=generation),
#             ],
#         }
# 
#     def validate_node(state: AgentState) -> Dict[str, Any]:
#         """验证节点（简化版）"""
#         documents = state.get("documents", [])
#         
#         return {
#             "relevance_score": "yes" if documents else "no",
#             "hallucination_score": "grounded",
#         }
# 
#     # 构建 LangGraph
#     graph = StateGraph(AgentState)
#     
#     graph.add_node("retrieve", retrieve_node)
#     graph.add_node("generate", generate_node)
#     graph.add_node("validate", validate_node)
#     
#     graph.set_entry_point("retrieve")
#     graph.add_edge("retrieve", "generate")
#     graph.add_edge("generate", "validate")
#     
#     # 编译图
#     app = graph.compile(checkpointer=memory.get_checkpointer())
# 
#     # 运行 Agent
#     initial_state: AgentState = {
#         "input": "用户的技术偏好是什么？",
#         "chat_history": [],
#         "documents": [],
#         "document_metadata": [],
#         "generation": "",
#         "relevance_score": "",
#         "hallucination_score": "",
#         "retry_count": 0,
#     }
# 
#     result = app.invoke(initial_state)
# 
#     print(f"生成结果: {result['generation']}")
#     print(f"相关性评分: {result['relevance_score']}")
#     print(f"幻觉评分: {result['hallucination_score']}")
# 
#     # 保存对话到记忆
#     memory.save_context(
#         inputs={"input": initial_state["input"]},
#         outputs={"generation": result["generation"]},
#     )
# 
# 
# def example_memory_management():
#     """记忆管理示例"""
#     memory = RAGEnhancedAgentMemory(
#         vector_db="qdrant",
#         short_term_threshold=5,
#         session_id="session_001",
#     )
# 
#     # 添加多轮对话
#     for i in range(6):
#         memory.save_context(
#             inputs={"input": f"问题{i}"},
#             outputs={"generation": f"回答{i}"},
#         )
# 
#     # 查看短期记忆统计
#     stats = memory.get_stats()
#     print(f"短期记忆统计: {stats['short_term']}")
# 
#     # 归档短期记忆到长期记忆
#     archived_ids = memory.archive_short_term_to_long_term()
#     print(f"归档了 {len(archived_ids)} 条记忆")
# 
#     # 获取 LLM 上下文
#     context = memory.get_context_for_llm(max_tokens=1000)
#     print(f"\nLLM 上下文:\n{context[:200]}...")
# 
# 
# def example_hybrid_retrieval():
#     """混合检索示例"""
#     memory = RAGEnhancedAgentMemory(
#         vector_db="qdrant",
#         use_hybrid_retrieval=True,
#         use_rerank=True,
#         session_id="session_001",
#     )
# 
#     # 添加一些记忆
#     memory.add_memory("Python是一种编程语言")
#     memory.add_memory("Java是另一种编程语言")
#     memory.add_memory("用户喜欢Python")
# 
#     # 使用混合检索
#     results = memory.search(
#         query="编程语言",
#         top_k=5,
#         use_hybrid=True,
#     )
# 
#     print(f"混合检索结果: {len(results)} 条")
#     for result in results:
#         print(
#             f"  - [{result['source']}] "
#             f"{result['content'][:50]}... "
#             f"(评分: {result['score']:.2f})"
#         )
# 
# 
# if __name__ == "__main__":
#     print("=== 基础使用示例 ===")
#     example_basic_usage()
# 
#     print("\n=== LangGraph 集成示例 ===")
#     example_langgraph_integration()
# 
#     print("\n=== 记忆管理示例 ===")
#     example_memory_management()
# 
#     print("\n=== 混合检索示例 ===")
#     example_hybrid_retrieval()
