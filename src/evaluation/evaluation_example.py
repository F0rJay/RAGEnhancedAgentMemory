"""
评估体系使用示例

展示如何使用 Ragas 评估和 Needle-in-a-Haystack 测试。
"""

# 注意：此文件仅作为示例，实际使用时需要安装依赖

# from src.evaluation.ragas_eval import RagasEvaluator
# from src.evaluation.needle_test import NeedleInHaystackTester
# 
# 
# def example_ragas_evaluation():
#     """Ragas 评估示例"""
#     # 初始化评估器
#     evaluator = RagasEvaluator()
# 
#     # 准备评估数据
#     questions = [
#         "什么是 Python？",
#         "用户的技术偏好是什么？",
#         "什么是机器学习？",
#     ]
# 
#     answers = [
#         "Python是一种高级编程语言",
#         "用户喜欢使用 Python 和 Django 框架",
#         "机器学习是人工智能的一个分支",
#     ]
# 
#     contexts = [
#         ["Python是一种编程语言", "Python由Guido van Rossum创建"],
#         ["用户喜欢Python", "用户使用Django构建Web应用"],
#         ["机器学习是AI的子领域", "ML使用算法从数据中学习"],
#     ]
# 
#     ground_truths = [
#         "Python是一种高级编程语言，由Guido van Rossum创建",
#         "用户的技术偏好是Python编程语言和Django Web框架",
#         "机器学习是人工智能的一个分支，使用算法从数据中学习",
#     ]
# 
#     # 执行评估
#     result = evaluator.evaluate(
#         questions=questions,
#         answers=answers,
#         contexts=contexts,
#         ground_truths=ground_truths,
#     )
# 
#     print("Ragas 评估结果:")
#     print(f"  上下文召回率 (Context Recall): {result.context_recall:.3f}")
#     print(f"  上下文精确度 (Context Precision): {result.context_precision:.3f}")
#     print(f"  忠实度 (Faithfulness): {result.faithfulness:.3f}")
#     print(f"  答案相关性 (Answer Relevancy): {result.answer_relevancy:.3f}")
#     print(f"  总体评分 (Overall Score): {result.overall_score:.3f}")
# 
#     # 检查阈值
#     checks = evaluator.check_threshold(result)
#     print("\n阈值检查:")
#     for metric, passed in checks.items():
#         status = "✓" if passed else "✗"
#         print(f"  {status} {metric}: {passed}")
# 
# 
# def example_needle_test():
#     """Needle-in-a-Haystack 测试示例"""
#     # 初始化测试器
#     tester = NeedleInHaystackTester(
#         token_estimate=20000,
#         needle_positions=[0.0, 0.25, 0.5, 0.75, 1.0],
#     )
# 
#     # 定义 Agent 函数（示例）
#     def agent_fn(haystack: str) -> str:
#         """模拟 Agent 函数"""
#         # 在实际应用中，这里会调用实际的 Agent
#         # 例如：memory.search(haystack) 或 LLM.generate(haystack)
#         if "秘密启动密码是 9527" in haystack:
#             return "9527"
#         return "未找到关键信息"
# 
#     # 运行单个测试
#     needle = "系统的秘密启动密码是 9527"
#     result = tester.run_test(
#         agent_fn=agent_fn,
#         needle=needle,
#         position=0.5,
#     )
# 
#     print(f"Needle 测试结果:")
#     print(f"  位置: {result.position}")
#     print(f"  成功: {result.success}")
#     print(f"  准确度: {result.accuracy:.3f}")
#     print(f"  期望答案: {result.expected_answer}")
#     print(f"  实际答案: {result.answer}")
# 
# 
# def example_needle_test_suite():
#     """Needle 测试套件示例"""
#     tester = NeedleInHaystackTester()
# 
#     # 定义 Agent 函数
#     def agent_fn(haystack: str) -> str:
#         """模拟 Agent 函数"""
#         # 实际应用中调用真实的 Agent
#         if "秘密密码" in haystack:
#             return "找到关键信息"
#         return "未找到"
# 
#     # 准备 Needles
#     needles = [
#         "系统的秘密启动密码是 9527",
#         "用户最喜欢的颜色是蓝色",
#         "项目的API密钥是 abc123",
#     ]
# 
#     # 运行测试套件
#     results = tester.run_test_suite(
#         agent_fn=agent_fn,
#         needles=needles,
#         positions=[0.0, 0.5, 1.0],
#     )
# 
#     # 分析结果
#     analysis = tester.analyze_results(results)
# 
#     print("Needle 测试套件结果:")
#     print(f"  总测试数: {analysis['total_tests']}")
#     print(f"  成功测试数: {analysis['successful_tests']}")
#     print(f"  成功率: {analysis['success_rate']:.2%}")
#     print(f"  平均准确度: {analysis['average_accuracy']:.3f}")
# 
#     print("\n按位置统计:")
#     for position, stats in analysis['position_stats'].items():
#         print(
#             f"  {position}: "
#             f"成功={stats['success_count']}/{stats['count']} "
#             f"({stats['success_rate']:.2%})"
#         )
# 
# 
# def example_single_evaluation():
#     """单个评估示例"""
#     evaluator = RagasEvaluator()
# 
#     result = evaluator.evaluate_single(
#         question="什么是 Python？",
#         answer="Python是一种高级编程语言",
#         contexts=[
#             "Python是一种编程语言",
#             "Python由Guido van Rossum创建",
#         ],
#         ground_truth="Python是一种高级编程语言，由Guido van Rossum创建",
#     )
# 
#     print("单个评估结果:")
#     print(f"  上下文召回率: {result.context_recall:.3f}")
#     print(f"  忠实度: {result.faithfulness:.3f}")
#     print(f"  总体评分: {result.overall_score:.3f}")
# 
# 
# def example_batch_evaluation():
#     """批量评估示例"""
#     evaluator = RagasEvaluator()
# 
#     evaluation_data = [
#         {
#             "question": "问题1",
#             "answer": "答案1",
#             "contexts": ["上下文1"],
#             "ground_truth": "标准答案1",
#         },
#         {
#             "question": "问题2",
#             "answer": "答案2",
#             "contexts": ["上下文2"],
#         },
#     ]
# 
#     result = evaluator.evaluate_batch(evaluation_data)
# 
#     print(f"批量评估结果: 总体评分={result.overall_score:.3f}")
# 
# 
# if __name__ == "__main__":
#     print("=== Ragas 评估示例 ===")
#     example_ragas_evaluation()
# 
#     print("\n=== Needle 测试示例 ===")
#     example_needle_test()
# 
#     print("\n=== Needle 测试套件示例 ===")
#     example_needle_test_suite()
# 
#     print("\n=== 单个评估示例 ===")
#     example_single_evaluation()
# 
#     print("\n=== 批量评估示例 ===")
#     example_batch_evaluation()
