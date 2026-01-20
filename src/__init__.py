"""
RAGEnhancedAgentMemory - 企业级 RAG 增强型 Agent 记忆系统

基于 LangGraph 与 vLLM 的生产化架构，实现分层记忆管理与自适应检索路由。
"""

__version__ = "0.1.0"
__author__ = "F0rJay"

from .core import RAGEnhancedAgentMemory

__all__ = [
    "__version__",
    "__author__",
    "RAGEnhancedAgentMemory",
]
