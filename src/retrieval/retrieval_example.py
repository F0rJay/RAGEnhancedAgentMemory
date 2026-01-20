"""
检索增强使用示例

展示如何使用混合检索和重排序功能。
"""

# 注意：此文件仅作为示例，实际使用时需要安装依赖和启动向量数据库

# from src.retrieval.reranker import Reranker
# from src.retrieval.hybrid import HybridRetriever
# from src.memory.long_term import LongTermMemory
# 
# 
# def example_reranker():
#     """重排序示例"""
#     # 初始化重排序器
#     reranker = Reranker(
#         model_name="BAAI/bge-reranker-large",
#         top_k=5,
#     )
# 
#     query = "什么是 Python 编程语言？"
#     documents = [
#         "Python是一种高级编程语言，由Guido van Rossum创建",
#         "Python有简洁的语法和丰富的库生态系统",
#         "Java是另一种编程语言",
#         "Python广泛用于数据科学和机器学习",
#         "C++是系统编程语言",
#     ]
# 
#     scores = [0.9, 0.85, 0.6, 0.8, 0.5]
# 
#     # 执行重排序
#     reranked = reranker.rerank(
#         query=query,
#         documents=documents,
#         scores=scores,
#     )
# 
#     print(f"重排序结果（Top {len(reranked)}）:")
#     for i, result in enumerate(reranked, 1):
#         print(
#             f"{i}. [原始: {result.original_score:.2f}, "
#             f"重排: {result.rerank_score:.2f}, "
#             f"变化: {result.rank_change}] "
#             f"{result.content[:50]}..."
#         )
# 
# 
# def example_hybrid_retrieval():
#     """混合检索示例"""
#     # 初始化长期记忆和检索器
#     long_memory = LongTermMemory(vector_db="qdrant")
#     reranker = Reranker(model_name="BAAI/bge-reranker-large")
# 
#     # 创建混合检索器
#     hybrid_retriever = HybridRetriever(
#         long_term_memory=long_memory,
#         reranker=reranker,
#         vector_weight=0.7,
#         keyword_weight=0.3,
#         use_rerank=True,
#     )
# 
#     # 执行混合检索
#     query = "用户的技术偏好是什么？"
#     results = hybrid_retriever.retrieve(
#         query=query,
#         top_k=10,
#         rerank_top_k=5,
#         session_id="session_001",
#     )
# 
#     print(f"\n混合检索结果（Top {len(results)}）:")
#     for i, result in enumerate(results, 1):
#         print(
#             f"{i}. [向量: {result.vector_score:.2f}, "
#             f"关键词: {result.keyword_score:.2f}, "
#             f"综合: {result.combined_score:.2f}, "
#             f"来源: {result.source}]\n"
#             f"   {result.content[:100]}...\n"
#         )
# 
# 
# def example_keyword_extraction():
#     """关键词提取示例"""
#     long_memory = LongTermMemory(vector_db="qdrant")
#     retriever = HybridRetriever(long_term_memory=long_memory)
# 
#     queries = [
#         "Python 编程语言的特点",
#         "什么是机器学习？",
#         "用户喜欢使用什么技术栈？",
#     ]
# 
#     for query in queries:
#         keywords = retriever._extract_keywords(query, expand=True)
#         print(f"查询: {query}")
#         print(f"关键词: {', '.join(keywords)}\n")
# 
# 
# def example_retrieval_with_fallback():
#     """带降级策略的检索示例"""
#     long_memory = LongTermMemory(vector_db="qdrant")
#     retriever = HybridRetriever(long_term_memory=long_memory)
# 
#     query = "测试查询"
# 
#     # 使用降级策略检索
#     results = retriever.retrieve_with_fallback(
#         query=query,
#         top_k=5,
#         session_id="session_001",
#     )
# 
#     print(f"检索结果: {len(results)} 条")
#     for result in results:
#         print(f"  - [{result.source}] {result.content[:50]}...")
# 
# 
# def example_rerank_batch():
#     """批量重排序示例"""
#     reranker = Reranker(model_name="BAAI/bge-reranker-large")
# 
#     query = "技术栈选择"
#     document_batches = [
#         ["Python是一种语言", "Java是另一种语言"],
#         ["机器学习很流行", "深度学习是子集"],
#     ]
# 
#     results = reranker.rerank_batch(
#         query=query,
#         document_batches=document_batches,
#         top_k=1,
#     )
# 
#     print("批量重排序结果:")
#     for i, batch_results in enumerate(results):
#         print(f"批次 {i+1}:")
#         for result in batch_results:
#             print(f"  - {result.content} (评分: {result.rerank_score:.2f})")
# 
# 
# if __name__ == "__main__":
#     print("=== 重排序示例 ===")
#     example_reranker()
# 
#     print("\n=== 混合检索示例 ===")
#     example_hybrid_retrieval()
# 
#     print("\n=== 关键词提取示例 ===")
#     example_keyword_extraction()
# 
#     print("\n=== 带降级策略的检索示例 ===")
#     example_retrieval_with_fallback()
# 
#     print("\n=== 批量重排序示例 ===")
#     example_rerank_batch()
