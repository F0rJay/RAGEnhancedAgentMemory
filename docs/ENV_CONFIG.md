# 环境变量配置说明

本文档详细说明如何填写 `.env` 文件中的各项配置。

## 📋 快速开始

```bash
# 1. 复制示例文件
cp env.example .env

# 2. 编辑 .env 文件，填写必要的配置
nano .env  # 或使用其他编辑器
```

## 🔧 配置项说明

### 1. 向量数据库配置

#### Qdrant 配置

```bash
# Qdrant 服务地址（本地部署默认端口）
QDRANT_URL=http://localhost:6333

# Qdrant API 密钥（可选）
# - 本地部署：通常为空
# - 云端部署：从 Qdrant Cloud 获取
QDRANT_API_KEY=

# 集合名称
QDRANT_COLLECTION_NAME=agent_memory
```

**如何填写**：
- **本地 Qdrant**：
  ```bash
  # 使用 Docker 启动
  docker run -p 6333:6333 qdrant/qdrant
  # QDRANT_API_KEY 留空即可
  ```
- **Qdrant Cloud**：
  ```bash
  # 从 https://cloud.qdrant.io/ 获取
  QDRANT_URL=https://your-cluster.qdrant.io
  QDRANT_API_KEY=your_api_key_here
  ```

#### Chroma 配置

```bash
# Chroma 服务地址和端口
CHROMA_HOST=localhost
CHROMA_PORT=8000
CHROMA_COLLECTION_NAME=agent_memory
```

**如何填写**：
- **本地 Chroma**：使用默认值即可
- **Chroma 服务**：根据实际部署地址修改

#### 向量数据库选择

```bash
# 选择使用的向量数据库：qdrant 或 chroma
VECTOR_DB=qdrant
```

---

### 2. 关系型数据库配置（PostgreSQL）

```bash
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=agent_memory
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password  # ⚠️ 必须填写
```

**如何填写**：

> **重要**：PostgreSQL 密码不是"获取"的，而是需要**自己设置**的！
> 
> 详细说明请参考：[PostgreSQL 配置指南](POSTGRESQL_SETUP.md)

**快速方法（使用 Docker，推荐）**：
```bash
# 1. 启动 PostgreSQL 容器并设置密码
docker run --name postgres-memory \
  -e POSTGRES_PASSWORD=your_password \
  -e POSTGRES_DB=agent_memory \
  -p 5432:5432 \
  -d postgres:15

# 2. 在 .env 文件中填写
POSTGRES_PASSWORD=your_password  # 这里填写你刚才设置的密码
```

**其他方式**：
- **本地 PostgreSQL**：安装时设置的密码，或使用重置密码方法
- **云端 PostgreSQL**：在创建数据库实例时设置的密码
- **已安装但忘记密码**：参考 [PostgreSQL 配置指南](POSTGRESQL_SETUP.md) 重置

**如果还没有安装 PostgreSQL**：
- 推荐使用 Docker 方式（最简单）
- 或参考 [PostgreSQL 配置指南](POSTGRESQL_SETUP.md) 安装

---

### 3. 模型配置

#### 嵌入模型

```bash
# 嵌入模型（用于向量化文本）
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5

# 向量维度（通常由模型决定，1024 适用于 bge-large）
EMBEDDING_DIM=1024

# 运行设备：cuda 或 cpu
EMBEDDING_DEVICE=cuda  # 有 GPU 时使用 cuda，否则使用 cpu
```

**常用嵌入模型**：
- `BAAI/bge-large-en-v1.5` - 英文模型（推荐）
- `BAAI/bge-large-zh-v1.5` - 中文模型
- `sentence-transformers/all-MiniLM-L6-v2` - 轻量级模型

#### 重排序模型

```bash
RERANK_MODEL=BAAI/bge-reranker-large
RERANK_TOP_K=5
```

**常用重排序模型**：
- `BAAI/bge-reranker-large` - 高性能（推荐）
- `BAAI/bge-reranker-base` - 更快的速度

#### vLLM 配置

```bash
# vLLM 模型路径（如果使用本地推理）
VLLM_MODEL_PATH=/path/to/model  # ⚠️ 需要填写实际路径

# GPU 内存使用率（0.0-1.0）
VLLM_GPU_MEMORY_UTILIZATION=0.9

# 最大模型长度
VLLM_MAX_MODEL_LEN=4096
```

**如何填写**：
- **本地模型（推荐）**：
  ```bash
  # 使用项目文件夹中的本地 Qwen 模型
  VLLM_MODEL_PATH=./models/Qwen2.5-14B-Instruct
  
  # 或使用绝对路径
  VLLM_MODEL_PATH=/root/autodl-tmp/RAGEnhancedAgentMemory/models/Qwen2.5-14B-Instruct
  ```
- **使用 HuggingFace 模型**：
  ```bash
  # 直接使用模型 ID（vLLM 会自动下载）
  VLLM_MODEL_PATH=Qwen/Qwen2.5-14B-Instruct
  ```
- **不使用本地推理**：
  ```bash
  # 留空，使用云端 API
  VLLM_MODEL_PATH=
  ```

---

### 4. API 密钥配置（可选）

#### OpenAI API

```bash
# OpenAI API 密钥（如果需要使用 OpenAI 模型）
OPENAI_API_KEY=  # ⚠️ 可选，从 https://platform.openai.com/api-keys 获取
```

**如何填写**：
- 访问 https://platform.openai.com/api-keys
- 创建新的 API 密钥
- 复制并粘贴到 `.env` 文件

#### Anthropic API

```bash
# Anthropic API 密钥（如果需要使用 Claude 模型）
ANTHROPIC_API_KEY=  # ⚠️ 可选，从 https://console.anthropic.com/ 获取
```

**如何填写**：
- 访问 https://console.anthropic.com/
- 创建 API 密钥
- 复制并粘贴到 `.env` 文件

---

### 5. 记忆系统配置

```bash
# 短期记忆最大轮数阈值
SHORT_TERM_THRESHOLD=10

# 长期记忆检索触发阈值（相关性评分 0-1）
LONG_TERM_TRIGGER=0.7

# 重排序 Top-K 数量
RERANK_TOP_K=5
```

**推荐值**：
- `SHORT_TERM_THRESHOLD`：10-20（根据对话长度调整）
- `LONG_TERM_TRIGGER`：0.6-0.8（越高越容易触发长期检索）
- `RERANK_TOP_K`：3-10（平衡精度和性能）

---

### 6. LangGraph 配置

```bash
# 检查点目录
CHECKPOINT_DIR=./checkpoints

# 是否启用检查点
ENABLE_CHECKPOINTING=true
```

**如何填写**：
- `CHECKPOINT_DIR`：设置保存状态的目录路径
- `ENABLE_CHECKPOINTING`：`true` 启用，`false` 禁用

---

### 7. 日志配置

```bash
# 日志级别：DEBUG, INFO, WARNING, ERROR
LOG_LEVEL=INFO

# 日志文件路径
LOG_FILE=./logs/agent_memory.log
```

**如何填写**：
- `LOG_LEVEL`：根据调试需要调整
  - `DEBUG` - 详细调试信息
  - `INFO` - 一般信息（推荐）
  - `WARNING` - 只显示警告和错误
- `LOG_FILE`：设置日志文件路径

---

### 8. 性能配置

```bash
# 异步批处理大小
ASYNC_BATCH_SIZE=32

# 最大并发请求数
MAX_CONCURRENT_REQUESTS=10
```

**如何调整**：
- 根据服务器资源调整
- GPU 内存充足时可以增加
- CPU 环境建议减小

---

## 📝 最小化配置示例

如果只是快速测试，可以只填写必要配置：

```bash
# 向量数据库（使用 Qdrant 本地部署）
VECTOR_DB=qdrant
QDRANT_URL=http://localhost:6333

# 模型配置（使用默认模型）
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
EMBEDDING_DEVICE=cpu  # 如果没有 GPU

# 记忆系统配置（使用默认值）
SHORT_TERM_THRESHOLD=10
LONG_TERM_TRIGGER=0.7

# 其他可以暂时留空
```

---

## ✅ 配置检查清单

- [ ] 向量数据库服务已启动（Qdrant 或 Chroma）
- [ ] PostgreSQL 已安装并运行（如使用）
- [ ] 嵌入模型路径正确或可从 HuggingFace 下载
- [ ] 如有 GPU，已安装 CUDA Toolkit
- [ ] API 密钥已填写（如使用云端服务）
- [ ] 日志目录已创建：`mkdir -p logs checkpoints`

---

## 🔒 安全建议

1. **不要提交 `.env` 文件到 Git**
   - `.env` 已在 `.gitignore` 中
   - 只提交 `env.example`

2. **保护敏感信息**
   - API 密钥和密码不要泄露
   - 生产环境使用密钥管理服务

3. **使用环境变量**
   - 生产环境可以设置系统环境变量
   - 而不是使用 `.env` 文件

---

## 🆘 常见问题

### Q: QDRANT_API_KEY 必须填写吗？
A: 本地部署留空即可，云端部署需要填写。

### Q: POSTGRES_PASSWORD 应该填什么？
A: 填写你设置 PostgreSQL 时的密码，如果是首次安装可以设置一个强密码。

### Q: VLLM_MODEL_PATH 怎么填？
A: 如果有本地模型，填写模型路径；否则留空，使用云端 API。

### Q: 没有 GPU 怎么办？
A: 设置 `EMBEDDING_DEVICE=cpu`，`VLLM_MODEL_PATH` 留空，使用云端 API。

---

## 📚 更多帮助

- [快速开始指南](../QUICKSTART.md)
- [使用文档](USAGE.md)
- [项目 README](../README.md)
