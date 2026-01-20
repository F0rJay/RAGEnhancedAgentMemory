"""
短期记忆使用示例

展示如何使用 ShortTermMemory 管理会话上下文。
"""

# 注意：此文件仅作为示例，实际使用时需要安装依赖

# from src.memory.short_term import ShortTermMemory
# from langchain_core.messages import HumanMessage, AIMessage
# 
# 
# def example_basic_usage():
#     """基础使用示例"""
#     # 创建短期记忆管理器
#     memory = ShortTermMemory(
#         max_turns=10,  # 最多保留10轮对话
#         compression_threshold=15,  # 超过15轮时触发压缩
#     )
# 
#     # 添加对话轮次
#     memory.add_conversation_turn(
#         human_message="你好，我是张三",
#         ai_message="你好张三！很高兴认识你。"
#     )
# 
#     memory.add_conversation_turn(
#         human_message="我的年龄是25岁",
#         ai_message="好的，你今年25岁。"
#     )
# 
#     # 获取最近上下文
#     context = memory.get_recent_context(n_turns=2)
#     print(f"对话轮数: {context['turn_count']}")
#     print(f"对话内容: {context['conversation']}")
# 
#     # 获取格式化的上下文（用于 LLM）
#     llm_context = memory.get_context_for_llm(max_tokens=1000)
#     print(f"\nLLM 上下文:\n{llm_context}")
# 
#     # 查看统计信息
#     stats = memory.get_stats()
#     print(f"\n统计信息: {stats}")
# 
# 
# def example_with_langchain_messages():
#     """与 LangChain 消息集成示例"""
#     memory = ShortTermMemory(max_turns=5)
# 
#     # 从 LangChain 消息添加对话
#     messages = [
#         HumanMessage(content="用户问题1"),
#         AIMessage(content="AI回答1"),
#         HumanMessage(content="用户问题2"),
#         AIMessage(content="AI回答2"),
#     ]
# 
#     memory.add_messages(messages)
# 
#     # 获取上下文
#     context = memory.get_recent_context()
#     print(f"添加了 {context['turn_count']} 轮对话")
# 
# 
# def example_compression():
#     """压缩示例"""
#     memory = ShortTermMemory(
#         max_turns=10,
#         compression_threshold=5,  # 降低阈值以便演示
#     )
# 
#     # 添加大量对话
#     for i in range(15):
#         memory.add_conversation_turn(
#             human_message=f"问题{i}",
#             ai_message=f"回答{i}"
#         )
# 
#     print(f"压缩前: {len(memory.conversation_buffer)} 轮对话")
# 
#     # 手动触发压缩
#     memory.compress(method="summary")
# 
#     print(f"压缩后: {len(memory.conversation_buffer)} 轮对话")
# 
# 
# def example_entity_tracking():
#     """实体跟踪示例"""
#     memory = ShortTermMemory(max_turns=10)
# 
#     # 添加包含实体的对话
#     memory.add_conversation_turn(
#         human_message='我想了解"Python"编程语言',
#         ai_message="Python是一种高级编程语言"
#     )
# 
#     memory.add_conversation_turn(
#         human_message='"Python"有哪些特性？',
#         ai_message="Python有简洁的语法和丰富的库"
#     )
# 
#     # 查看关键实体
#     context = memory.get_recent_context(include_entities=True)
#     if "key_entities" in context:
#         print("关键实体:")
#         for entity in context["key_entities"][:5]:
#             print(f"  - {entity['name']}: 重要性={entity['importance_score']:.2f}")
# 
# 
# if __name__ == "__main__":
#     print("=== 基础使用示例 ===")
#     example_basic_usage()
# 
#     print("\n=== LangChain 集成示例 ===")
#     example_with_langchain_messages()
# 
#     print("\n=== 压缩示例 ===")
#     example_compression()
# 
#     print("\n=== 实体跟踪示例 ===")
#     example_entity_tracking()
