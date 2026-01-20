"""
自适应路由使用示例

展示如何使用 MemoryRouter 进行智能记忆路由决策。
"""

# 注意：此文件仅作为示例，实际使用时需要安装依赖

# from src.memory.routing import MemoryRouter, RoutingContext, RouteDecision
# from src.memory.short_term import ShortTermMemory
# from src.memory.long_term import LongTermMemory
# 
# 
# def example_basic_routing():
#     """基础路由示例"""
#     # 初始化记忆系统和路由器
#     short_memory = ShortTermMemory(max_turns=10)
#     long_memory = LongTermMemory(vector_db="qdrant")
#     router = MemoryRouter(
#         short_term_threshold=10,
#         long_term_trigger=0.7,
#     )
# 
#     # 添加一些对话到短期记忆
#     short_memory.add_conversation_turn(
#         human_message="我喜欢Python编程",
#         ai_message="Python是一种很好的编程语言"
#     )
# 
#     # 创建路由上下文
#     context = RoutingContext(
#         query="Python有哪些特性？",
#         short_term_memory=short_memory,
#         long_term_memory=long_memory,
#         session_id="session_001",
#     )
# 
#     # 执行路由决策
#     result = router.route(context)
# 
#     print(f"路由决策: {result.decision.value}")
#     print(f"推理: {result.reasoning}")
#     print(f"置信度: {result.confidence:.2f}")
# 
#     if result.short_term_context:
#         print(f"\n短期记忆: {result.short_term_context['turn_count']} 轮对话")
# 
#     if result.long_term_results:
#         print(f"\n长期记忆: 检索到 {len(result.long_term_results)} 条记忆")
# 
# 
# def example_empty_short_term():
#     """短期记忆为空的路由示例"""
#     short_memory = ShortTermMemory(max_turns=10)
#     long_memory = LongTermMemory(vector_db="qdrant")
#     router = MemoryRouter()
# 
#     context = RoutingContext(
#         query="用户偏好是什么？",
#         short_term_memory=short_memory,
#         long_term_memory=long_memory,
#     )
# 
#     result = router.route(context)
# 
#     # 短期记忆为空，应该使用长期记忆
#     assert result.decision == RouteDecision.LONG_TERM_ONLY
#     print(f"决策: {result.decision.value} - {result.reasoning}")
# 
# 
# def example_archive_trigger():
#     """触发归档的路由示例"""
#     short_memory = ShortTermMemory(max_turns=5)  # 降低阈值以便测试
#     long_memory = LongTermMemory(vector_db="qdrant")
#     router = MemoryRouter(short_term_threshold=5)
# 
#     # 添加多轮对话，超过阈值
#     for i in range(6):
#         short_memory.add_conversation_turn(
#             human_message=f"问题{i}",
#             ai_message=f"回答{i}"
#         )
# 
#     context = RoutingContext(
#         query="总结一下之前的对话",
#         short_term_memory=short_memory,
#         long_term_memory=long_memory,
#         session_id="session_001",
#     )
# 
#     result = router.route(context)
# 
#     # 应该触发归档
#     assert result.decision == RouteDecision.ARCHIVE_AND_RETRIEVE
#     assert result.should_archive is True
#     print(f"决策: {result.decision.value}")
#     print(f"是否归档: {result.should_archive}")
# 
# 
# def example_low_relevance_trigger():
#     """低相关性触发的路由示例"""
#     short_memory = ShortTermMemory(max_turns=10)
#     long_memory = LongTermMemory(vector_db="qdrant")
#     router = MemoryRouter(long_term_trigger=0.7)
# 
#     # 添加对话
#     short_memory.add_conversation_turn(
#         human_message="今天天气不错",
#         ai_message="是的，适合外出"
#     )
# 
#     context = RoutingContext(
#         query="用户的技术栈是什么？",  # 与当前对话不相关
#         short_term_memory=short_memory,
#         long_term_memory=long_memory,
#         recent_relevance_score=0.5,  # 低相关性
#         session_id="session_001",
#     )
# 
#     result = router.route(context)
# 
#     # 低相关性应该触发混合检索
#     assert result.decision == RouteDecision.HYBRID
#     print(f"决策: {result.decision.value}")
#     print(f"推理: {result.reasoning}")
# 
# 
# def example_complex_query_routing():
#     """复杂查询路由示例"""
#     short_memory = ShortTermMemory(max_turns=10)
#     long_memory = LongTermMemory(vector_db="qdrant")
#     router = MemoryRouter()
# 
#     context = RoutingContext(
#         query="为什么之前的对话中提到的问题没有得到解决？请分析一下原因并解释差异。",
#         short_term_memory=short_memory,
#         long_term_memory=long_memory,
#         session_id="session_001",
#     )
# 
#     result = router.route(context)
# 
#     # 复杂查询应该使用混合检索
#     print(f"查询复杂度评估")
#     complexity = router._assess_query_complexity(context.query)
#     print(f"复杂度: {complexity:.2f}")
#     print(f"路由决策: {result.decision.value}")
# 
# 
# def example_routing_summary():
#     """路由摘要示例"""
#     short_memory = ShortTermMemory(max_turns=10)
#     long_memory = LongTermMemory(vector_db="qdrant")
#     router = MemoryRouter()
# 
#     # 添加对话
#     short_memory.add_conversation_turn(
#         human_message="测试问题",
#         ai_message="测试回答"
#     )
# 
#     context = RoutingContext(
#         query="查询测试",
#         short_term_memory=short_memory,
#         long_term_memory=long_memory,
#     )
# 
#     result = router.route(context)
#     summary = router.get_routing_summary(result)
# 
#     print("路由摘要:")
#     for key, value in summary.items():
#         print(f"  {key}: {value}")
# 
# 
# if __name__ == "__main__":
#     print("=== 基础路由示例 ===")
#     example_basic_routing()
# 
#     print("\n=== 短期记忆为空的路由示例 ===")
#     example_empty_short_term()
# 
#     print("\n=== 触发归档的路由示例 ===")
#     example_archive_trigger()
# 
#     print("\n=== 低相关性触发的路由示例 ===")
#     example_low_relevance_trigger()
# 
#     print("\n=== 复杂查询路由示例 ===")
#     example_complex_query_routing()
# 
#     print("\n=== 路由摘要示例 ===")
#     example_routing_summary()
