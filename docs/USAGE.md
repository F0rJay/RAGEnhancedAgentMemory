# 使用文档

详细的使用指南和最佳实践。

## 📖 目录

1. [基础概念](#基础概念)
2. [核心功能](#核心功能)
3. [高级用法](#高级用法)
4. [最佳实践](#最佳实践)
5. [故障排除](#故障排除)

## 基础概念

### 三层记忆架构

1. **瞬时记忆**：当前请求的中间状态，存在于 LangGraph State 中
2. **短期记忆**：最近 N 轮对话，使用滑动窗口管理
3. **长期记忆**：向量数据库中的语义记忆，支持异步归档

### 工作流程

```
用户输入 → 路由决策 → 检索上下文 → 生成回答 → 保存记忆
```

## 核心功能

### 1. 记忆管理

```python
from src.core import RAGEnhancedAgentMemory

memory = RAGEnhancedAgentMemory(vector_db="qdrant")

# 添加记忆
memory_id = memory.add_memory(
    content="重要信息",
    metadata={"category": "preference"},
)

# 搜索记忆
results = memory.search("查询文本", top_k=5)
```

### 2. LangGraph 集成

```python
from langgraph.graph import StateGraph
from src.graph.state import AgentState

graph = StateGraph(AgentState)
graph.add_node("retrieve", memory.retrieve_context)
graph.add_node("generate", generate_node)
graph.add_node("validate", validate_node)

app = graph.compile(checkpointer=memory.get_checkpointer())
```

### 3. 自适应路由

系统会自动决策：
- 何时使用短期记忆
- 何时从长期记忆检索
- 是否需要混合检索

### 4. 混合检索

```python
# 自动使用混合检索（向量 + 关键词）
results = memory.search(
    query="查询文本",
    top_k=5,
    use_hybrid=True,
)
```

## 高级用法

### 自定义配置

```python
memory = RAGEnhancedAgentMemory(
    vector_db="qdrant",
    embedding_model="custom-model",
    rerank_model="custom-reranker",
    short_term_threshold=15,
    long_term_trigger=0.8,
    use_hybrid_retrieval=True,
    use_rerank=True,
)
```

### 手动归档

```python
# 手动将短期记忆归档到长期记忆
archived_ids = memory.archive_short_term_to_long_term()
```

### 评估系统质量

```python
from src.evaluation import RagasEvaluator

evaluator = RagasEvaluator()
result = evaluator.evaluate(
    questions=["问题1", "问题2"],
    answers=["答案1", "答案2"],
    contexts=[["上下文1"], ["上下文2"]],
)
```

## 最佳实践

### 1. 会话管理

为每个用户或对话分配唯一的 `session_id`：

```python
memory = RAGEnhancedAgentMemory(
    session_id=f"user_{user_id}_{timestamp}",
)
```

### 2. 记忆归档策略

- 定期归档短期记忆（达到阈值时自动触发）
- 使用元数据标记记忆类型和重要性
- 定期清理过期的长期记忆

### 3. 检索优化

- 根据查询类型选择合适的检索策略
- 使用混合检索提升召回率
- 使用重排序提升精度

### 4. 性能优化

- 批量处理记忆添加
- 使用异步归档减少阻塞
- 合理设置短期记忆阈值

## 故障排除

### 问题 1: 向量数据库连接失败

**解决方案**：
1. 检查数据库服务是否运行
2. 验证 `.env` 文件中的连接配置
3. 检查网络连接和防火墙设置

### 问题 2: 模型加载失败

**解决方案**：
1. 确保有足够的磁盘空间
2. 检查网络连接（模型会自动下载）
3. 可以手动下载模型到指定路径

### 问题 3: 检索结果不理想

**解决方案**：
1. 调整检索参数（top_k, score_threshold）
2. 使用混合检索
3. 启用重排序
4. 优化嵌入模型选择

### 问题 4: 内存占用过高

**解决方案**：
1. 降低短期记忆阈值
2. 定期清理短期记忆
3. 压缩长期记忆
4. 使用更小的嵌入模型

## 📚 更多资源

- [API 参考](API.md) (待完善)
- [架构设计](ARCHITECTURE.md) (待完善)
- [性能调优](PERFORMANCE.md) (待完善)
