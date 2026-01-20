"""
长期记忆使用示例

展示如何使用 LongTermMemory 进行记忆归档和检索。
"""

# 注意：此文件仅作为示例，实际使用时需要安装依赖和启动向量数据库

# from src.memory.long_term import LongTermMemory, MemoryType
# from src.memory.short_term import ShortTermMemory
# 
# 
# def example_basic_usage():
#     """基础使用示例"""
#     # 创建长期记忆管理器
#     memory = LongTermMemory(
#         vector_db="qdrant",  # 或 "chroma"
#         embedding_model="BAAI/bge-large-en-v1.5",
#         collection_name="agent_memory",
#     )
# 
#     # 添加记忆
#     memory_id = memory.add_memory(
#         content="用户喜欢喝咖啡，每天早晨都会来一杯",
#         metadata={"category": "preference"},
#         session_id="session_001",
#     )
#     print(f"添加记忆，ID: {memory_id}")
# 
#     # 检索记忆
#     results = memory.search(
#         query="用户喜欢什么饮料？",
#         top_k=3,
#     )
# 
#     print(f"\n检索到 {len(results)} 条记忆:")
#     for i, result in enumerate(results, 1):
#         print(f"{i}. [分数: {result.relevance_score:.3f}] {result.content[:50]}...")
# 
# 
# def example_archive_from_short_term():
#     """从短期记忆归档示例"""
#     # 创建短期和长期记忆管理器
#     short_memory = ShortTermMemory(max_turns=10)
#     long_memory = LongTermMemory(vector_db="qdrant")
# 
#     # 添加一些对话到短期记忆
#     short_memory.add_conversation_turn(
#         human_message="我喜欢Python编程",
#         ai_message="Python是一种很好的编程语言"
#     )
# 
#     short_memory.add_conversation_turn(
#         human_message="我的项目使用Django框架",
#         ai_message="Django是Python的Web框架"
#     )
# 
#     # 获取短期记忆上下文
#     context = short_memory.get_recent_context()
# 
#     # 归档到长期记忆
#     archived_ids = long_memory.archive_from_short_term(
#         conversation_turns=context["conversation"],
#         session_id="session_001",
#         deduplicate=True,
#     )
# 
#     print(f"归档了 {len(archived_ids)} 条记忆到长期记忆库")
# 
# 
# def example_batch_operations():
#     """批量操作示例"""
#     memory = LongTermMemory(vector_db="qdrant")
# 
#     # 批量添加记忆
#     contents = [
#         "用户的工作时间是早上9点到下午6点",
#         "用户使用的操作系统是 Linux",
#         "用户最喜欢的编程语言是 Python",
#     ]
# 
#     metadatas = [
#         {"category": "schedule"},
#         {"category": "system"},
#         {"category": "preference"},
#     ]
# 
#     memory_ids = memory.add_memories_batch(
#         contents=contents,
#         metadatas=metadatas,
#         session_id="session_001",
#     )
# 
#     print(f"批量添加了 {len(memory_ids)} 条记忆")
# 
# 
# def example_search_with_filters():
#     """带过滤条件的检索示例"""
#     memory = LongTermMemory(vector_db="qdrant")
# 
#     # 检索特定会话的记忆
#     results = memory.search(
#         query="用户偏好",
#         top_k=5,
#         session_id="session_001",
#         score_threshold=0.7,
#     )
# 
#     print(f"检索到 {len(results)} 条相关记忆")
# 
#     # 使用元数据过滤
#     results = memory.search(
#         query="工作时间",
#         top_k=5,
#         metadata_filter={"category": "schedule"},
#     )
# 
#     print(f"过滤后的结果: {len(results)} 条")
# 
# 
# def example_get_stats():
#     """获取统计信息示例"""
#     memory = LongTermMemory(vector_db="qdrant")
# 
#     stats = memory.get_stats()
#     print("长期记忆统计信息:")
#     for key, value in stats.items():
#         print(f"  {key}: {value}")
# 
# 
# if __name__ == "__main__":
#     print("=== 基础使用示例 ===")
#     example_basic_usage()
# 
#     print("\n=== 从短期记忆归档示例 ===")
#     example_archive_from_short_term()
# 
#     print("\n=== 批量操作示例 ===")
#     example_batch_operations()
# 
#     print("\n=== 带过滤条件的检索示例 ===")
#     example_search_with_filters()
# 
#     print("\n=== 统计信息示例 ===")
#     example_get_stats()
