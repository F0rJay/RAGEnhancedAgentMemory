"""
增强系统测试脚本

测试 RAG 增强型 Agent 记忆系统的性能，与基线系统进行对比。
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.core import RAGEnhancedAgentMemory

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# 延迟导入 RAGEnhancedAgentMemory，避免在依赖缺失时无法导入模块
from scripts.benchmark.conversation_dataset import (
    ConversationDatasetGenerator,
    TestDataset,
)

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class EnhancedTestRunner:
    """增强系统测试运行器"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        初始化测试运行器
        
        Args:
            output_dir: 结果输出目录
        """
        self.output_dir = output_dir or Path(__file__).parent.parent / "benchmark_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"初始化增强系统测试运行器，输出目录: {self.output_dir}")
    
    def run_test(
        self,
        dataset: TestDataset,
        memory_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        运行单个测试
        
        Args:
            dataset: 测试数据集
            memory_config: 记忆系统配置参数
        
        Returns:
            测试结果
        """
        # 延迟导入，避免在模块级别导入时缺少依赖
        try:
            from src.core import RAGEnhancedAgentMemory
        except ImportError as e:
            raise ImportError(
                f"无法导入 RAGEnhancedAgentMemory。请确保已安装所有依赖：{e}"
            ) from e
        
        logger.info(f"开始测试增强系统，数据集: {dataset.name}")
        
        # 创建记忆系统
        config = memory_config or {}
        memory = RAGEnhancedAgentMemory(
            vector_db=config.get("vector_db", "qdrant"),
            short_term_threshold=config.get("short_term_threshold", 10),
            long_term_trigger=config.get("long_term_trigger", 0.7),
            use_hybrid_retrieval=config.get("use_hybrid_retrieval", True),
            use_rerank=config.get("use_rerank", True),
            session_id=f"test_{dataset.name}_{int(time.time())}",
        )
        
        # 运行对话
        retrieval_latencies = []
        generation_latencies = []
        context_lengths = []
        storage_sizes = []
        successful_early_access = []
        
        # 模拟 LLM（简化版）
        from scripts.benchmark.baseline_agents import SimpleMockLLM
        llm = SimpleMockLLM(delay=0.1)
        
        for conv in dataset.conversations:
            # 保存对话到记忆系统
            memory.save_context(
                inputs={"input": conv.human_message},
                outputs={"generation": conv.ai_message},
            )
            
            # 如果包含关键信息，也添加到长期记忆
            if conv.contains_key_info:
                memory.add_memory(
                    content=conv.ai_message,
                    metadata={
                        "turn_number": conv.turn_number,
                        "key_info_type": conv.key_info_type,
                    },
                )
        
        # 运行测试问题
        test_results = []
        for test_q in dataset.test_questions:
            # 检索上下文
            start_time = time.time()
            state = {
                "input": test_q["question"],
                "chat_history": [],
                "documents": [],
                "document_metadata": [],
                "generation": "",
                "relevance_score": "",
                "hallucination_score": "",
                "retry_count": 0,
            }
            
            retrieval_result = memory.retrieve_context(state)
            retrieval_latency = time.time() - start_time
            retrieval_latencies.append(retrieval_latency)
            
            # 获取检索到的文档
            documents = retrieval_result.get("documents", [])
            context_length = sum(len(doc) for doc in documents) // 4  # 估算 token
            context_lengths.append(context_length)
            
            # 生成回答
            gen_start = time.time()
            response, gen_latency = llm.generate(test_q["question"], documents)
            generation_latencies.append(gen_latency)
            
            # 检查是否成功
            success = self._check_answer_correctness(
                documents, test_q["expected_key_info"], test_q["key_info_turn"]
            )
            successful_early_access.append(success)
            
            test_results.append({
                "question": test_q["question"],
                "expected_key_info": test_q["expected_key_info"],
                "key_info_turn": test_q["key_info_turn"],
                "success": success,
                "retrieval_latency": retrieval_latency,
                "generation_latency": gen_latency,
                "memory_type": retrieval_result.get("memory_type", "unknown"),
            })
        
        # 计算存储大小（估算）
        stats = memory.get_stats()
        # 这里需要实际统计存储，简化处理
        final_storage = self._estimate_storage_size(memory, dataset)
        
        # 计算指标
        avg_retrieval_latency = sum(retrieval_latencies) / len(retrieval_latencies) if retrieval_latencies else 0
        avg_generation_latency = sum(generation_latencies) / len(generation_latencies) if generation_latencies else 0
        avg_context_length = sum(context_lengths) / len(context_lengths) if context_lengths else 0
        success_rate = sum(successful_early_access) / len(successful_early_access) if successful_early_access else 0
        
        results = {
            "agent_type": "rag_enhanced",
            "dataset_name": dataset.name,
            "total_turns": dataset.total_turns,
            "memory_config": config,
            "metrics": {
                "success_rate": success_rate,
                "avg_retrieval_latency": avg_retrieval_latency,
                "avg_generation_latency": avg_generation_latency,
                "total_latency": sum(retrieval_latencies) + sum(generation_latencies),
                "avg_context_length": avg_context_length,
                "final_storage_bytes": final_storage,
                "max_context_length": max(context_lengths) if context_lengths else 0,
                "avg_storage_bytes": final_storage,  # 简化处理
            },
            "test_results": test_results,
            "timestamp": time.time(),
        }
        
        logger.info(
            f"测试完成: RAG增强系统, 成功率: {success_rate:.2%}, "
            f"平均延迟: {avg_retrieval_latency + avg_generation_latency:.3f}s"
        )
        
        return results
    
    def _check_answer_correctness(
        self,
        documents: List[str],
        expected_key_info: str,
        key_info_turn: int,
    ) -> bool:
        """检查回答是否正确"""
        # 检查文档是否包含关键信息
        context_text = "\n".join(documents).lower()
        key_parts = expected_key_info.lower().split(":")[-1].strip()
        
        for part in key_parts.split(","):
            part = part.strip()
            if part and part not in context_text:
                return False
        
        return True
    
    def _estimate_storage_size(
        self,
        memory: Any,  # RAGEnhancedAgentMemory
        dataset: TestDataset,
    ) -> int:
        """估算存储大小"""
        # 简化估算：短期记忆 + 长期记忆
        stats = memory.get_stats()
        short_term_size = stats.get("short_term", {}).get("turn_count", 0) * 200  # 每轮约200字节
        
        # 长期记忆估算（向量 + 元数据）
        # 简化处理：每轮对话约500字节（向量 + 文本）
        long_term_size = len([c for c in dataset.conversations if c.contains_key_info]) * 500
        
        return short_term_size + long_term_size
    
    def run_enhanced_suite(self) -> Dict[str, Any]:
        """运行完整的增强系统测试套件"""
        logger.info("开始运行增强系统测试套件...")
        
        # 生成测试数据集
        generator = ConversationDatasetGenerator(seed=42)  # 使用相同的seed确保一致性
        test_suites = generator.generate_standard_test_suite()
        
        # 记忆系统配置
        memory_config = {
            "vector_db": "qdrant",
            "short_term_threshold": 10,
            "long_term_trigger": 0.7,
            "use_hybrid_retrieval": True,
            "use_rerank": True,
        }
        
        all_results = {}
        
        # 运行所有测试
        for suite_name, dataset in test_suites.items():
            result = self.run_test(dataset, memory_config)
            all_results[suite_name] = result
        
        # 保存结果
        output_file = self.output_dir / "enhanced_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"增强系统测试完成，结果已保存到: {output_file}")
        
        # 打印摘要
        self._print_summary(all_results)
        
        return all_results
    
    def _print_summary(self, results: Dict[str, Any]) -> None:
        """打印测试结果摘要"""
        print("\n" + "=" * 80)
        print("增强系统测试结果摘要")
        print("=" * 80)
        
        for suite_name, result in results.items():
            metrics = result["metrics"]
            print(f"\n数据集: {suite_name} ({result['total_turns']} 轮)")
            print("-" * 80)
            print(f"  成功率: {metrics['success_rate']:.2%}")
            print(f"  平均检索延迟: {metrics['avg_retrieval_latency']:.3f}s")
            print(f"  平均生成延迟: {metrics['avg_generation_latency']:.3f}s")
            print(f"  最终存储: {metrics['final_storage_bytes'] / 1024:.2f} KB")
            print(f"  平均上下文长度: {metrics['avg_context_length']:.0f} tokens")
            print()


def main():
    """主函数"""
    print("=== RAG 增强系统性能测试 ===\n")
    
    runner = EnhancedTestRunner()
    results = runner.run_enhanced_suite()
    
    print("\n=== 测试完成 ===")
    print(f"详细结果已保存到: {runner.output_dir / 'enhanced_results.json'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)
        sys.exit(1)
