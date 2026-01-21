"""
Ragas 评估模块

集成 Ragas 框架，评估 RAG 系统的质量指标。
"""

import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field

try:
    from ragas import evaluate
    from ragas.metrics import (
        context_recall,
        context_precision,
        faithfulness,
        answer_relevancy,
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    evaluate = None
    context_recall = None
    context_precision = None
    faithfulness = None
    answer_relevancy = None
    Dataset = None

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from ..config import get_settings


@dataclass
class EvaluationResult:
    """评估结果"""
    context_recall: float
    context_precision: float
    faithfulness: float
    answer_relevancy: float
    overall_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class RagasEvaluator:
    """
    Ragas 评估器
    
    使用 Ragas 框架评估 RAG 系统的质量指标：
    - Context Recall（上下文召回率）
    - Context Precision（上下文精确度）
    - Faithfulness（忠实度）
    - Answer Relevancy（答案相关性）
    """

    def __init__(
        self,
        metrics: Optional[List[Any]] = None,
    ):
        """
        初始化 Ragas 评估器
        
        Args:
            metrics: 评估指标列表（None 表示使用默认指标）
        """
        if not RAGAS_AVAILABLE:
            raise ImportError(
                "Ragas 未安装。请运行: pip install ragas>=0.1.0"
            )

        # 默认评估指标
        if metrics is None:
            self.metrics = [
                context_recall,
                context_precision,
                faithfulness,
                answer_relevancy,
            ]
        else:
            self.metrics = metrics

        logger.info(
            f"Ragas 评估器初始化: {len(self.metrics)} 个评估指标"
        )

    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """
        评估 RAG 系统
        
        Args:
            questions: 问题列表
            answers: 答案列表
            contexts: 上下文列表（每个问题的上下文）
            ground_truths: 标准答案列表（可选，用于 Context Recall）
        
        Returns:
            评估结果
        """
        if not RAGAS_AVAILABLE:
            raise ImportError("Ragas 未安装")

        # 准备数据
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }

        if ground_truths:
            data["ground_truths"] = ground_truths

        # 创建数据集
        dataset = Dataset.from_dict(data)

        # 执行评估
        logger.info(f"开始评估: {len(questions)} 个样本")
        start_time = time.time()

        result = evaluate(
            dataset=dataset,
            metrics=self.metrics,
        )

        evaluation_time = time.time() - start_time

        # 提取评估指标
        metrics_dict = result.to_dict()

        context_recall_score = metrics_dict.get("context_recall", 0.0)
        context_precision_score = metrics_dict.get("context_precision", 0.0)
        faithfulness_score = metrics_dict.get("faithfulness", 0.0)
        answer_relevancy_score = metrics_dict.get("answer_relevancy", 0.0)

        # 计算总体评分（平均值）
        overall_score = (
            context_recall_score
            + context_precision_score
            + faithfulness_score
            + answer_relevancy_score
        ) / 4.0

        logger.info(
            f"评估完成: 耗时 {evaluation_time:.2f}s, "
            f"总体评分: {overall_score:.3f}"
        )

        return EvaluationResult(
            context_recall=context_recall_score,
            context_precision=context_precision_score,
            faithfulness=faithfulness_score,
            answer_relevancy=answer_relevancy_score,
            overall_score=overall_score,
            metadata={
                "evaluation_time": evaluation_time,
                "sample_count": len(questions),
                "metrics_used": [m.name for m in self.metrics],
            },
        )

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> EvaluationResult:
        """
        评估单个问答对
        
        Args:
            question: 问题
            answer: 答案
            contexts: 上下文列表
            ground_truth: 标准答案（可选）
        
        Returns:
            评估结果
        """
        return self.evaluate(
            questions=[question],
            answers=[answer],
            contexts=[contexts],
            ground_truths=[ground_truth] if ground_truth else None,
        )

    def evaluate_batch(
        self,
        evaluation_data: List[Dict[str, Any]],
    ) -> EvaluationResult:
        """
        批量评估
        
        Args:
            evaluation_data: 评估数据列表，每个元素包含：
                - question: 问题
                - answer: 答案
                - contexts: 上下文列表
                - ground_truth: 标准答案（可选）
        
        Returns:
            评估结果
        """
        questions = [item["question"] for item in evaluation_data]
        answers = [item["answer"] for item in evaluation_data]
        contexts = [item["contexts"] for item in evaluation_data]
        ground_truths = [
            item.get("ground_truth") for item in evaluation_data
            if "ground_truth" in item
        ]

        return self.evaluate(
            questions=questions,
            answers=answers,
            contexts=contexts,
            ground_truths=ground_truths if ground_truths else None,
        )

    def check_threshold(
        self,
        result: EvaluationResult,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, bool]:
        """
        检查评估结果是否满足阈值要求
        
        Args:
            result: 评估结果
            thresholds: 阈值字典，例如：
                {"context_recall": 0.8, "faithfulness": 0.9}
                None 表示使用默认阈值
        
        Returns:
            每个指标的达标情况
        """
        if thresholds is None:
            thresholds = {
                "context_recall": 0.7,
                "context_precision": 0.7,
                "faithfulness": 0.8,
                "answer_relevancy": 0.7,
                "overall_score": 0.75,
            }

        checks = {
            "context_recall": result.context_recall >= thresholds.get("context_recall", 0.7),
            "context_precision": result.context_precision >= thresholds.get("context_precision", 0.7),
            "faithfulness": result.faithfulness >= thresholds.get("faithfulness", 0.8),
            "answer_relevancy": result.answer_relevancy >= thresholds.get("answer_relevancy", 0.7),
            "overall_score": result.overall_score >= thresholds.get("overall_score", 0.75),
        }

        return checks
