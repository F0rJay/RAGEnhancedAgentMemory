"""
基线系统测试脚本

运行基线系统（滑动窗口和全量摘要）的性能测试，收集基准指标。
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from scripts.benchmark.baseline_agents import (
    SlidingWindowAgent,
    SummarizationAgent,
    SimpleMockLLM,
    BaselineMetrics,
    create_baseline_agent,
)
from scripts.benchmark.conversation_dataset import (
    ConversationDatasetGenerator,
    TestDataset,
)

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class BaselineTestRunner:
    """基线测试运行器"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        初始化测试运行器
        
        Args:
            output_dir: 结果输出目录
        """
        self.output_dir = output_dir or Path(__file__).parent.parent / "benchmark_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.llm = SimpleMockLLM(delay=0.1)  # 模拟 LLM（可替换为真实 LLM）
        
        logger.info(f"初始化基线测试运行器，输出目录: {self.output_dir}")
    
    def run_test(
        self,
        agent_type: str,
        dataset: TestDataset,
        agent_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        运行单个测试
        
        Args:
            agent_type: Agent 类型 ("sliding_window" 或 "summarization")
            dataset: 测试数据集
            agent_config: Agent 配置参数
        
        Returns:
            测试结果
        """
        logger.info(f"开始测试: {agent_type}, 数据集: {dataset.name}")
        
        # 创建 Agent
        config = agent_config or {}
        if agent_type == "sliding_window":
            agent = SlidingWindowAgent(window_size=config.get("window_size", 10))
        elif agent_type == "summarization":
            agent = SummarizationAgent(
                summary_threshold=config.get("summary_threshold", 15),
                compression_ratio=config.get("compression_ratio", 0.3),
            )
        else:
            raise ValueError(f"未知的 Agent 类型: {agent_type}")
        
        # 运行对话
        total_latency = 0.0
        retrieval_latencies = []
        generation_latencies = []
        context_lengths = []
        storage_sizes = []
        successful_early_access = []
        
        for conv in dataset.conversations:
            # 添加对话
            agent.add_conversation(conv.human_message, conv.ai_message)
            
            # 获取上下文（用于下一轮）
            if conv.turn_number < dataset.total_turns:
                context, metrics = agent.get_context("")
                retrieval_latencies.append(metrics.retrieval_latency)
                context_lengths.append(metrics.context_length)
                storage_sizes.append(metrics.storage_bytes)
        
        # 运行测试问题
        test_results = []
        for test_q in dataset.test_questions:
            # 获取上下文
            context, metrics = agent.get_context(test_q["question"])
            retrieval_latency = metrics.retrieval_latency
            
            # 生成回答
            response, gen_latency = self.llm.generate(test_q["question"], context)
            generation_latencies.append(gen_latency)
            
            # 检查是否成功（简化版：检查关键信息是否在上下文中）
            key_info = test_q["expected_key_info"]
            success = self._check_answer_correctness(
                context, key_info, test_q["key_info_turn"], agent
            )
            successful_early_access.append(success)
            
            total_latency += retrieval_latency + gen_latency
            
            test_results.append({
                "question": test_q["question"],
                "expected_key_info": key_info,
                "key_info_turn": test_q["key_info_turn"],
                "success": success,
                "retrieval_latency": retrieval_latency,
                "generation_latency": gen_latency,
            })
        
        # 计算最终存储大小
        final_storage = agent.get_storage_size()
        
        # 计算指标
        avg_retrieval_latency = sum(retrieval_latencies) / len(retrieval_latencies) if retrieval_latencies else 0
        avg_generation_latency = sum(generation_latencies) / len(generation_latencies) if generation_latencies else 0
        avg_context_length = sum(context_lengths) / len(context_lengths) if context_lengths else 0
        success_rate = sum(successful_early_access) / len(successful_early_access) if successful_early_access else 0
        
        results = {
            "agent_type": agent_type,
            "dataset_name": dataset.name,
            "total_turns": dataset.total_turns,
            "agent_config": config,
            "metrics": {
                "success_rate": success_rate,
                "avg_retrieval_latency": avg_retrieval_latency,
                "avg_generation_latency": avg_generation_latency,
                "total_latency": total_latency,
                "avg_context_length": avg_context_length,
                "final_storage_bytes": final_storage,
                "max_context_length": max(context_lengths) if context_lengths else 0,
                "avg_storage_bytes": sum(storage_sizes) / len(storage_sizes) if storage_sizes else 0,
            },
            "test_results": test_results,
            "timestamp": time.time(),
        }
        
        logger.info(
            f"测试完成: {agent_type}, 成功率: {success_rate:.2%}, "
            f"平均延迟: {avg_retrieval_latency + avg_generation_latency:.3f}s"
        )
        
        return results
    
    def _check_answer_correctness(
        self,
        context: List[str],
        expected_key_info: str,
        key_info_turn: int,
        agent: Any,
    ) -> bool:
        """
        检查回答是否正确
        
        Args:
            context: 上下文
            expected_key_info: 期望的关键信息
            key_info_turn: 关键信息出现的轮次
        
        Returns:
            是否正确
        """
        # 检查 Agent 是否能访问早期信息
        can_access = agent.can_access_early_info(key_info_turn)
        
        if not can_access:
            return False
        
        # 检查上下文是否包含关键信息（简化版）
        context_text = "\n".join(context).lower()
        key_parts = expected_key_info.lower().split(":")[-1].strip()  # 提取值部分
        
        # 简单匹配：检查关键部分是否在上下文中
        for part in key_parts.split(","):
            part = part.strip()
            if part and part not in context_text:
                return False
        
        return True
    
    def run_baseline_suite(self) -> Dict[str, Any]:
        """运行完整的基线测试套件"""
        logger.info("开始运行基线测试套件...")
        
        # 生成测试数据集
        generator = ConversationDatasetGenerator(seed=42)
        test_suites = generator.generate_standard_test_suite()
        
        # 测试配置
        agent_configs = {
            "sliding_window": {"window_size": 10},
            "summarization": {"summary_threshold": 15, "compression_ratio": 0.3},
        }
        
        all_results = {}
        
        # 运行所有测试组合
        for agent_type, config in agent_configs.items():
            agent_results = {}
            for suite_name, dataset in test_suites.items():
                result = self.run_test(agent_type, dataset, config)
                agent_results[suite_name] = result
            
            all_results[agent_type] = agent_results
        
        # 保存结果
        output_file = self.output_dir / "baseline_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"基线测试完成，结果已保存到: {output_file}")
        
        # 打印摘要
        self._print_summary(all_results)
        
        return all_results
    
    def _print_summary(self, results: Dict[str, Any]) -> None:
        """打印测试结果摘要"""
        print("\n" + "=" * 80)
        print("基线测试结果摘要")
        print("=" * 80)
        
        for agent_type, agent_results in results.items():
            print(f"\n【{agent_type}】")
            print("-" * 80)
            
            for suite_name, result in agent_results.items():
                metrics = result["metrics"]
                print(f"  数据集: {suite_name} ({result['total_turns']} 轮)")
                print(f"    成功率: {metrics['success_rate']:.2%}")
                print(f"    平均检索延迟: {metrics['avg_retrieval_latency']:.3f}s")
                print(f"    平均生成延迟: {metrics['avg_generation_latency']:.3f}s")
                print(f"    最终存储: {metrics['final_storage_bytes'] / 1024:.2f} KB")
                print(f"    平均上下文长度: {metrics['avg_context_length']:.0f} tokens")
                print()


def main():
    """主函数"""
    print("=== 基线系统性能测试 ===\n")
    
    runner = BaselineTestRunner()
    results = runner.run_baseline_suite()
    
    print("\n=== 测试完成 ===")
    print(f"详细结果已保存到: {runner.output_dir / 'baseline_results.json'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)
        sys.exit(1)
