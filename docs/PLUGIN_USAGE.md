# 插件使用指南

本指南说明如何将 RAGEnhancedAgentMemory 作为插件集成到其他项目中。

## 📦 安装

### 方式一：从 GitHub 安装（推荐）

```bash
pip install git+https://github.com/F0rJay/RAGEnhancedAgentMemory.git
```

### 方式二：从本地安装

```bash
git clone https://github.com/F0rJay/RAGEnhancedAgentMemory.git
cd RAGEnhancedAgentMemory
pip install -e .
```

### 方式三：安装可选依赖

```bash
# 安装 vLLM 支持（需要 CUDA）
pip install "rag-enhanced-agent-memory[vllm]"

# 安装开发依赖
pip install "rag-enhanced-agent-memory[dev]"
```

## 🚀 基础使用

### 导入模块

```python
from rag_enhanced_agent_memory import RAGEnhancedAgentMemory
```

### 初始化记忆系统

```python
memory = RAGEnhancedAgentMemory(
    vector_db="qdrant",  # 或 "chroma"
    embedding_model="BAAI/bge-large-en-v1.5",
    rerank_model="BAAI/bge-reranker-large",
    session_id="my_session_001"
)
```

### 核心功能

#### 1. 添加记忆

```python
# 添加长期记忆
memory_id = memory.add_memory(
    content="用户喜欢使用 Python 编程",
    metadata={"category": "preference", "topic": "programming"}
)
```

#### 2. 搜索记忆

```python
# 语义搜索
results = memory.search("用户的技术偏好", top_k=5)
for result in results:
    print(f"内容: {result['content']}")
    print(f"评分: {result['score']}")
```

#### 3. 保存对话上下文

```python
# 保存对话到短期记忆
memory.save_context(
    inputs={"input": "用户问题"},
    outputs={"generation": "AI 回答"}
)
```

#### 4. 检索上下文

```python
# 检索相关上下文（用于生成回答）
context = memory.retrieve_context(
    state={"input": "用户问题", "chat_history": []}
)
```

## 🔌 LangGraph 集成

### 创建 Agent 图

```python
from langgraph.graph import StateGraph
from rag_enhanced_agent_memory.graph.state import AgentState

# 初始化记忆系统
memory = RAGEnhancedAgentMemory(vector_db="qdrant")

# 创建图
graph = StateGraph(AgentState)
graph.add_node("retrieve", memory.retrieve_context)
graph.add_node("generate", your_generate_function)
graph.add_node("save", memory.save_context)

# 设置边
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", "save")

# 编译应用
app = graph.compile(checkpointer=memory.get_checkpointer())
```

### 运行 Agent

```python
# 运行 Agent
config = {"configurable": {"thread_id": memory.session_id}}
result = app.invoke(
    {"input": "用户问题", "chat_history": []},
    config=config
)
```

## ⚙️ 配置选项

### 环境变量配置

创建 `.env` 文件：

```bash
# 向量数据库
VECTOR_DB=qdrant
QDRANT_URL=http://localhost:6333

# 模型配置
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
RERANK_MODEL=BAAI/bge-reranker-large

# 记忆系统参数
SHORT_TERM_THRESHOLD=10
LONG_TERM_TRIGGER=0.7
```

### 代码配置

```python
memory = RAGEnhancedAgentMemory(
    vector_db="qdrant",
    embedding_model="BAAI/bge-large-en-v1.5",
    rerank_model="BAAI/bge-reranker-large",
    short_term_threshold=10,      # 短期记忆最大轮数
    long_term_trigger=0.7,        # 长期记忆触发阈值
    use_hybrid_retrieval=True,     # 使用混合检索
    use_rerank=True,               # 使用重排序
    checkpoint_dir="./checkpoints", # 检查点目录
    session_id="custom_session"     # 会话 ID
)
```

## 📝 完整示例

```python
from rag_enhanced_agent_memory import RAGEnhancedAgentMemory
from langgraph.graph import StateGraph
from rag_enhanced_agent_memory.graph.state import AgentState

# 1. 初始化记忆系统
memory = RAGEnhancedAgentMemory(
    vector_db="qdrant",
    session_id="demo_session"
)

# 2. 添加初始记忆
memory.add_memory("用户喜欢 Python 编程")
memory.add_memory("用户使用 Django 框架")

# 3. 创建 LangGraph Agent
def generate_node(state: AgentState) -> AgentState:
    # 使用检索到的上下文生成回答
    context = "\n".join(state.get("documents", []))
    # ... 调用 LLM 生成回答 ...
    state["generation"] = "生成的回答"
    return state

graph = StateGraph(AgentState)
graph.add_node("retrieve", memory.retrieve_context)
graph.add_node("generate", generate_node)
graph.add_node("save", memory.save_context)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", "save")

app = graph.compile(checkpointer=memory.get_checkpointer())

# 4. 运行 Agent
config = {"configurable": {"thread_id": memory.session_id}}
result = app.invoke(
    {"input": "用户的技术栈是什么？", "chat_history": []},
    config=config
)

print(f"回答: {result['generation']}")
```

## 🔧 故障排除

### 导入错误

如果遇到 `ModuleNotFoundError: No module named 'rag_enhanced_agent_memory'`：

1. 确保已正确安装包：
   ```bash
   pip install git+https://github.com/F0rJay/RAGEnhancedAgentMemory.git
   ```

2. 检查 Python 环境：
   ```bash
   python -c "import rag_enhanced_agent_memory; print('OK')"
   ```

### 向量数据库连接失败

1. 确保向量数据库服务已启动（Qdrant 或 Chroma）
2. 检查 `.env` 文件中的配置
3. 对于 Qdrant，可以使用本地嵌入式模式（无需 Docker）

### 模型下载慢

1. 配置 HuggingFace 镜像源
2. 使用预下载的模型路径
3. 参考 [镜像配置文档](MIRROR_SETUP.md)

## 📚 更多资源

- [完整文档](../README.md)
- [快速开始指南](../QUICKSTART.md)
- [API 文档](API.md)（待完善）
- [项目状态报告](PROJECT_STATUS.md)

## 🤝 获取帮助

如有问题，请提交 [Issue](https://github.com/F0rJay/RAGEnhancedAgentMemory/issues)
