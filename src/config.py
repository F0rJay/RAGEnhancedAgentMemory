"""
配置管理模块

使用 pydantic-settings 管理环境变量和应用配置。
"""

from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """应用配置类"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 向量数据库配置
    vector_db: Literal["qdrant", "chroma"] = Field(default="qdrant", description="向量数据库选择")

    # Qdrant 配置
    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant 服务地址")
    qdrant_api_key: Optional[str] = Field(default=None, description="Qdrant API 密钥")
    qdrant_collection_name: str = Field(default="agent_memory", description="Qdrant 集合名称")

    # Chroma 配置
    chroma_host: str = Field(default="localhost", description="Chroma 服务地址")
    chroma_port: int = Field(default=8000, description="Chroma 服务端口")
    chroma_collection_name: str = Field(default="agent_memory", description="Chroma 集合名称")

    # PostgreSQL 配置
    postgres_host: str = Field(default="localhost", description="PostgreSQL 主机")
    postgres_port: int = Field(default=5432, description="PostgreSQL 端口")
    postgres_db: str = Field(default="agent_memory", description="PostgreSQL 数据库名")
    postgres_user: str = Field(default="postgres", description="PostgreSQL 用户名")
    postgres_password: str = Field(default="your_password", description="PostgreSQL 密码")

    # 模型配置
    embedding_model: str = Field(
        default="BAAI/bge-large-en-v1.5", description="嵌入模型名称"
    )
    embedding_dim: int = Field(default=1024, description="嵌入向量维度")
    embedding_device: Literal["cuda", "cpu"] = Field(default="cuda", description="嵌入模型设备")

    rerank_model: str = Field(
        default="BAAI/bge-reranker-large", description="重排序模型名称"
    )
    rerank_top_k: int = Field(default=5, description="重排序 Top-K 数量")

    # vLLM 配置
    vllm_model_path: Optional[str] = Field(default=None, description="vLLM 模型路径")
    vllm_gpu_memory_utilization: float = Field(
        default=0.9, description="vLLM GPU 内存使用率"
    )
    vllm_max_model_len: int = Field(default=4096, description="vLLM 最大模型长度")

    # API 密钥
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API 密钥")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API 密钥")

    # 记忆系统配置
    short_term_threshold: int = Field(
        default=10, description="短期记忆最大轮数阈值"
    )
    long_term_trigger: float = Field(
        default=0.7, description="长期记忆检索触发阈值 (相关性评分)"
    )

    # LangGraph 配置
    checkpoint_dir: str = Field(default="./checkpoints", description="检查点目录")
    enable_checkpointing: bool = Field(default=True, description="是否启用检查点")

    # 日志配置
    log_level: str = Field(default="INFO", description="日志级别")
    log_file: str = Field(default="./logs/agent_memory.log", description="日志文件路径")

    # 性能配置
    async_batch_size: int = Field(default=32, description="异步批处理大小")
    max_concurrent_requests: int = Field(
        default=10, description="最大并发请求数"
    )


# 全局配置实例
settings = Settings()


def get_settings() -> Settings:
    """获取配置实例"""
    return settings
