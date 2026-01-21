"""
Needle-in-a-Haystack 测试模块

针对长上下文 Agent，验证其在长窗口下的检索能力。
"""

import random
import time
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class NeedleTestResult:
    """Needle 测试结果"""
    position: float  # Needle 位置（0.0-1.0）
    success: bool  # 是否成功检索到 Needle
    answer: str  # Agent 的回答
    expected_answer: str  # 期望的答案
    accuracy: float  # 准确度评分（0-1）
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class NeedleInHaystackTester:
    """
    Needle-in-a-Haystack 测试器
    
    在长上下文（Haystack）中插入关键事实（Needle），
    验证 Agent 在不同位置都能准确检索和回答。
    """

    def __init__(
        self,
        haystack_template: Optional[str] = None,
        token_estimate: int = 20000,
        needle_positions: Optional[List[float]] = None,
    ):
        """
        初始化 Needle 测试器
        
        Args:
            haystack_template: 无关文本模板（None 表示使用默认模板）
            token_estimate: 估计的 token 数量
            needle_positions: Needle 插入位置列表（None 表示使用默认位置）
        """
        self.haystack_template = haystack_template or self._default_haystack_template()
        self.token_estimate = token_estimate
        self.needle_positions = needle_positions or [0.0, 0.25, 0.5, 0.75, 1.0]

        logger.info(
            f"Needle 测试器初始化: "
            f"token_estimate={token_estimate}, "
            f"positions={self.needle_positions}"
        )

    def _default_haystack_template(self) -> str:
        """默认无关文本模板"""
        # 生成一个包含大量无关文本的模板
        paragraphs = []
        for i in range(100):
            paragraphs.append(
                f"这是第 {i+1} 段无关文本。"
                f"这段文本包含了各种不相关的信息，"
                f"用于模拟长上下文环境。"
                f"在实际应用中，这些文本可能是文档、"
                f"历史记录或其他类型的上下文内容。"
            )
        return "\n\n".join(paragraphs)

    def generate_haystack(
        self,
        needle: str,
        position: float,
        haystack_size: Optional[int] = None,
    ) -> str:
        """
        生成包含 Needle 的 Haystack
        
        Args:
            needle: 关键事实（Needle）
            position: 插入位置（0.0-1.0）
            haystack_size: Haystack 大小（字符数，None 表示使用默认）
        
        Returns:
            包含 Needle 的 Haystack 文本
        """
        haystack = self.haystack_template

        if haystack_size:
            # 调整 Haystack 大小
            current_size = len(haystack)
            if haystack_size > current_size:
                # 扩展 Haystack
                haystack += haystack * ((haystack_size - current_size) // current_size)
                haystack = haystack[:haystack_size]
            elif haystack_size < current_size:
                # 截断 Haystack
                haystack = haystack[:haystack_size]

        # 计算插入位置（字符索引）
        insert_index = int(len(haystack) * position)

        # 插入 Needle
        haystack_with_needle = (
            haystack[:insert_index]
            + f"\n\n[关键信息] {needle} [关键信息]\n\n"
            + haystack[insert_index:]
        )

        return haystack_with_needle

    def generate_test_cases(
        self,
        needles: List[str],
        positions: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        生成测试用例
        
        Args:
            needles: Needle 列表
            positions: 插入位置列表（None 表示使用初始化时的位置）
        
        Returns:
            测试用例列表
        """
        if positions is None:
            positions = self.needle_positions

        test_cases = []
        for needle in needles:
            for position in positions:
                haystack = self.generate_haystack(needle, position)
                test_cases.append({
                    "needle": needle,
                    "position": position,
                    "haystack": haystack,
                    "question": f"关键信息是什么？请从上下文中提取关键事实。",
                    "expected_answer": needle,
                })

        logger.info(f"生成了 {len(test_cases)} 个测试用例")
        return test_cases

    def run_test(
        self,
        agent_fn: Callable[[str], str],
        needle: str,
        position: float,
        question: Optional[str] = None,
    ) -> NeedleTestResult:
        """
        运行单个 Needle 测试
        
        Args:
            agent_fn: Agent 函数，接受 Haystack 文本，返回回答
            needle: 关键事实（Needle）
            position: 插入位置（0.0-1.0）
            question: 问题（None 表示使用默认问题）
        
        Returns:
            测试结果
        """
        if question is None:
            question = f"关键信息是什么？请从上下文中提取关键事实。"

        # 生成 Haystack
        haystack = self.generate_haystack(needle, position)

        # 执行 Agent
        try:
            answer = agent_fn(haystack)
        except Exception as e:
            logger.error(f"Agent 执行失败: {e}")
            answer = ""

        # 评估答案
        success, accuracy = self._evaluate_answer(answer, needle)

        return NeedleTestResult(
            position=position,
            success=success,
            answer=answer,
            expected_answer=needle,
            accuracy=accuracy,
            metadata={
                "haystack_length": len(haystack),
                "question": question,
            },
        )

    def run_test_suite(
        self,
        agent_fn: Callable[[str], str],
        needles: List[str],
        positions: Optional[List[float]] = None,
    ) -> List[NeedleTestResult]:
        """
        运行测试套件
        
        Args:
            agent_fn: Agent 函数
            needles: Needle 列表
            positions: 插入位置列表
        
        Returns:
            测试结果列表
        """
        if positions is None:
            positions = self.needle_positions

        results = []
        for needle in needles:
            for position in positions:
                logger.info(f"运行测试: needle={needle[:30]}..., position={position}")
                result = self.run_test(agent_fn, needle, position)
                results.append(result)

        return results

    def _evaluate_answer(
        self,
        answer: str,
        expected: str,
    ) -> Tuple[bool, float]:
        """
        评估答案
        
        Args:
            answer: Agent 的回答
            expected: 期望的答案
        
        Returns:
            (是否成功, 准确度评分)
        """
        # 简单的字符串匹配评估（实际应用中可以使用更复杂的评估方法）
        answer_lower = answer.lower().strip()
        expected_lower = expected.lower().strip()

        # 精确匹配
        if expected_lower in answer_lower or answer_lower in expected_lower:
            return True, 1.0

        # 部分匹配
        # 计算字符重叠度
        common_chars = set(answer_lower) & set(expected_lower)
        total_chars = set(answer_lower) | set(expected_lower)
        if total_chars:
            similarity = len(common_chars) / len(total_chars)
            success = similarity > 0.7
            return success, similarity

        return False, 0.0

    def analyze_results(
        self,
        results: List[NeedleTestResult],
    ) -> Dict[str, Any]:
        """
        分析测试结果
        
        Args:
            results: 测试结果列表
        
        Returns:
            分析结果
        """
        if not results:
            return {}

        total = len(results)
        successful = sum(1 for r in results if r.success)
        success_rate = successful / total if total > 0 else 0.0

        avg_accuracy = sum(r.accuracy for r in results) / total if total > 0 else 0.0

        # 按位置分析
        position_stats = {}
        for position in self.needle_positions:
            position_results = [r for r in results if abs(r.position - position) < 0.01]
            if position_results:
                pos_success = sum(1 for r in position_results if r.success)
                pos_success_rate = pos_success / len(position_results)
                position_stats[f"position_{position}"] = {
                    "count": len(position_results),
                    "success_count": pos_success,
                    "success_rate": pos_success_rate,
                }

        analysis = {
            "total_tests": total,
            "successful_tests": successful,
            "success_rate": success_rate,
            "average_accuracy": avg_accuracy,
            "position_stats": position_stats,
        }

        logger.info(
            f"测试结果分析: 总数={total}, "
            f"成功率={success_rate:.2%}, "
            f"平均准确度={avg_accuracy:.3f}"
        )

        return analysis
