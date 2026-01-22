"""
RAGEnhancedAgentMemory - RAG 增强型 Agent 记忆系统

基于 LangGraph 与 vLLM 的高性能架构，实现分层记忆管理与自适应检索路由。
"""

__version__ = "0.1.0"
__author__ = "F0rJay"

# 使用 lazy import 避免模块级导入 heavy 组件（如 torch）
# 这样可以确保用户 import src 时不会触发 torch 导入
# 只有用户真正访问 RAGEnhancedAgentMemory 时才导入
def __getattr__(name: str):
    """
    延迟导入 RAGEnhancedAgentMemory，避免模块级导入触发 torch
    
    支持用法：
        from src import RAGEnhancedAgentMemory
        from src.core import RAGEnhancedAgentMemory  # 也支持
    """
    if name == "RAGEnhancedAgentMemory":
        from .core import RAGEnhancedAgentMemory
        return RAGEnhancedAgentMemory
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "__version__",
    "__author__",
    "RAGEnhancedAgentMemory",
]
