"""
重排序模块

使用重排序模型对检索结果进行重新排序，提升检索精度。
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# 延迟导入 CrossEncoder 和 torch，避免模块级导入触发 torch
# 这些导入将在 Reranker._init_model 方法中按需导入
CrossEncoder = None  # 占位符，实际导入在 _init_model 方法中
TRANSFORMERS_AVAILABLE = None  # 占位符，实际检测在 _init_model 方法中

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from ..config import get_settings


@dataclass
class RerankedResult:
    """重排序结果"""
    content: str
    original_score: float  # 原始相关性评分
    rerank_score: float  # 重排序评分
    metadata: Dict[str, Any]
    rank_change: int  # 排名变化（负数表示提升，正数表示下降）


class Reranker:
    """
    重排序器
    
    使用交叉编码器（CrossEncoder）对检索结果进行重新排序，
    提升检索精度和上下文质量。
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        top_k: Optional[int] = None,
        device: Optional[str] = None,
    ):
        """
        初始化重排序器
        
        Args:
            model_name: 重排序模型名称
            top_k: 返回 Top-K 结果
            device: 设备 (cuda/cpu)
        """
        settings = get_settings()
        self.model_name = model_name or settings.rerank_model
        self.top_k = top_k or settings.rerank_top_k

        if device is None:
            device = "cuda" if settings.embedding_device == "cuda" else "cpu"
        self.device = device

        # 初始化模型
        logger.info(f"加载重排序模型: {self.model_name}")
        self.model = None
        self.tokenizer = None
        self._init_model()

        logger.info(f"重排序器初始化完成: model={self.model_name}, device={self.device}")

    def _init_model(self) -> None:
        """初始化重排序模型（延迟导入，避免模块级导入触发 torch）"""
        try:
            # 延迟导入，避免模块级导入触发 torch
            # 优先尝试使用 sentence-transformers 的 CrossEncoder
            try:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(
                    self.model_name,
                    max_length=512,
                    device=self.device,
                )
                self.use_cross_encoder = True
                logger.info("使用 CrossEncoder 重排序模型")
                return
            except ImportError:
                # CrossEncoder 不可用，尝试使用 transformers
                pass
            
            # 使用 transformers 库作为备选方案
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                import torch
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name
                )
                self.model.to(self.device)
                self.model.eval()
                self.use_cross_encoder = False
                logger.info("使用 Transformers 重排序模型")
                return
            except ImportError:
                # transformers 也不可用
                pass
            
            # 如果两种方案都失败，抛出错误
            raise ImportError(
                "重排序模型库未安装。请运行: "
                "pip install sentence-transformers>=2.3.0 或 transformers>=4.36.0"
            )
        except Exception as e:
            logger.error(f"重排序模型加载失败: {e}")
            raise

    def rerank(
        self,
        query: str,
        documents: List[str],
        scores: Optional[List[float]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        top_k: Optional[int] = None,
    ) -> List[RerankedResult]:
        """
        对检索结果进行重排序
        
        Args:
            query: 查询文本
            documents: 文档列表
            scores: 原始相关性评分列表
            metadatas: 元数据列表
            top_k: 返回 Top-K 结果（None 表示使用初始化时的值）
        
        Returns:
            重排序后的结果列表
        """
        if not documents:
            return []

        top_k = top_k or self.top_k

        # 记录原始排名
        original_ranks = list(range(len(documents)))

        # 执行重排序
        start_time = time.time()
        rerank_scores = self._compute_rerank_scores(query, documents)
        rerank_time = time.time() - start_time

        # 创建结果列表（包含原始排名和重排序评分）
        results = []
        for i, (doc, orig_score, rerank_score, metadata) in enumerate(
            zip(
                documents,
                scores or [0.0] * len(documents),
                rerank_scores,
                metadatas or [{}] * len(documents),
            )
        ):
            results.append({
                "index": i,
                "content": doc,
                "original_score": orig_score,
                "rerank_score": float(rerank_score),
                "metadata": metadata,
            })

        # 按重排序评分排序
        results.sort(key=lambda x: x["rerank_score"], reverse=True)

        # 转换为 RerankedResult 并计算排名变化
        reranked_results = []
        for new_rank, result in enumerate(results[:top_k]):
            original_rank = result["index"]
            rank_change = original_rank - new_rank  # 负数表示提升

            reranked_results.append(
                RerankedResult(
                    content=result["content"],
                    original_score=result["original_score"],
                    rerank_score=result["rerank_score"],
                    metadata=result["metadata"],
                    rank_change=rank_change,
                )
            )

        logger.debug(
            f"重排序完成: {len(documents)} -> {len(reranked_results)} 个结果, "
            f"耗时: {rerank_time:.3f}s"
        )

        return reranked_results

    def _compute_rerank_scores(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:
        """
        计算重排序评分
        
        Args:
            query: 查询文本
            documents: 文档列表
        
        Returns:
            评分列表
        """
        if self.use_cross_encoder:
            # 使用 CrossEncoder
            pairs = [[query, doc] for doc in documents]
            scores = self.model.predict(pairs, show_progress_bar=False)
            # CrossEncoder 返回的是 logits，需要转换为概率
            if len(scores.shape) == 1:
                # 如果是二分类，取正类概率
                import numpy as np
                # 简单的 sigmoid 转换
                scores = 1 / (1 + np.exp(-scores))
            return scores.tolist()
        else:
            # 使用 Transformers
            import torch
            scores_list = []

            for doc in documents:
                # 编码查询和文档对
                inputs = self.tokenizer(
                    query,
                    doc,
                    truncation=True,
                    max_length=512,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # 计算评分
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # 获取正类 logit（假设二分类模型）
                    if hasattr(outputs, "logits"):
                        logits = outputs.logits
                        if logits.dim() > 1:
                            logits = logits[:, 0]  # 取第一个 logit
                        # 转换为概率（简单的 sigmoid）
                        score = torch.sigmoid(logits).item()
                    else:
                        score = outputs[0].item()
                    scores_list.append(score)

            return scores_list

    def rerank_batch(
        self,
        query: str,
        document_batches: List[List[str]],
        scores_batches: Optional[List[List[float]]] = None,
        metadatas_batches: Optional[List[List[Dict[str, Any]]]] = None,
        top_k: Optional[int] = None,
    ) -> List[List[RerankedResult]]:
        """
        批量重排序
        
        Args:
            query: 查询文本
            document_batches: 文档批次列表
            scores_batches: 评分批次列表
            metadatas_batches: 元数据批次列表
            top_k: 返回 Top-K 结果
        
        Returns:
            重排序结果批次列表
        """
        results = []
        for i, documents in enumerate(document_batches):
            scores = scores_batches[i] if scores_batches else None
            metadatas = metadatas_batches[i] if metadatas_batches else None

            reranked = self.rerank(
                query=query,
                documents=documents,
                scores=scores,
                metadatas=metadatas,
                top_k=top_k,
            )
            results.append(reranked)

        return results
