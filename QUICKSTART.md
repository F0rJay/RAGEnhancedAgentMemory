# 快速开始指南

本指南将帮助您快速上手 RAGEnhancedAgentMemory。

## 📋 前置要求

- Python >= 3.9
- CUDA >= 11.8 (如需使用 GPU)
- 8GB+ RAM (推荐 16GB+)

## 🚀 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/F0rJay/RAGEnhancedAgentMemory.git
cd RAGEnhancedAgentMemory
```

### 2. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

```bash
cp env.example .env
# 编辑 .env 文件，填入必要的配置
```

### 5. 启动向量数据库（可选）

#### Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

#### Chroma

```bash
pip install chromadb
chroma run --path ./chroma_db
```

## 💡 快速示例

### 示例 1: 基础使用

```python
from src.core import RAGEnhancedAgentMemory

# 初始化记忆系统
memory = RAGEnhancedAgentMemory(
    vector_db="qdrant",
    session_id="my_session",
)

# 添加记忆
memory.add_memory("用户喜欢Python编程")

# 搜索记忆
results = memory.search("用户的技术偏好", top_k=5)
for result in results:
    print(result['content'])
```

### 示例 2: LangGraph 集成

```python
from src.core import RAGEnhancedAgentMemory
from src.graph.state import AgentState
from langgraph.graph import StateGraph

# 初始化
memory = RAGEnhancedAgentMemory(vector_db="qdrant")

# 构建图
graph = StateGraph(AgentState)
graph.add_node("retrieve", memory.retrieve_context)
# ... 添加其他节点

app = graph.compile(checkpointer=memory.get_checkpointer())
result = app.invoke({"input": "用户问题"})
```

### 示例 3: 运行脚本示例

```bash
# 基础示例
python scripts/basic_example.py

# LangGraph 集成示例
python scripts/langgraph_example.py

# 评估示例
python scripts/evaluation_example.py
```

## 📚 更多资源

- [完整文档](README.md)
- [API 文档](docs/API.md) (待完善)
- [示例代码](src/)
- [测试用例](tests/)

## 🐛 常见问题

### Q: 导入错误怎么办？

A: 确保已安装所有依赖：`pip install -r requirements.txt`

### Q: 向量数据库连接失败？

A: 确保向量数据库服务已启动，并检查 `.env` 文件中的配置

### Q: 模型下载慢？

A: 可以配置镜像源或使用预下载的模型路径

## 🔗 相关链接

- [GitHub 仓库](https://github.com/F0rJay/RAGEnhancedAgentMemory)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [Ragas 文档](https://docs.ragas.io/)

## 📞 获取帮助

如有问题，请提交 [Issue](https://github.com/F0rJay/RAGEnhancedAgentMemory/issues)
