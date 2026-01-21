# Docker 配置指南

## Docker 安装状态

Docker 已通过 `apt` 安装在系统中：
- Docker 版本: 28.2.2
- Docker Compose 版本: 1.29.2

## 当前限制

由于运行环境可能不支持 systemd，Docker daemon 可能无法直接启动。

## 解决方案

### 方案 1: 使用 Qdrant 本地嵌入式模式（推荐，无需 Docker）

项目已配置为优先使用 Qdrant 的本地嵌入式模式，无需启动 Docker 或向量数据库服务器。

**优点：**
- 无需 Docker
- 无需额外服务器
- 数据持久化在本地
- 开箱即用
- 高性能

**使用方法：**
```bash
# 设置环境变量使用 Qdrant（默认配置）
export VECTOR_DB=qdrant

# 运行项目
python scripts/verify_functionality.py
```

数据将自动保存在 `./qdrant_db` 目录中。

### 方案 2: 使用 Chroma 本地持久化模式（无需 Docker）

项目也支持 Chroma 的本地持久化模式。

**优点：**
- 无需 Docker
- 无需额外服务器
- 数据持久化在本地
- 开箱即用

**使用方法：**
```bash
# 设置环境变量使用 Chroma
export VECTOR_DB=chroma
export CHROMA_HOST=localhost
export CHROMA_PORT=8000

# 运行项目
python scripts/verify_functionality.py
```

数据将自动保存在 `./chroma_db` 目录中。

### 方案 3: 启动 Docker 和 Qdrant 服务器（如果环境支持）

如果需要使用 Qdrant 作为向量数据库：

```bash
# 1. 启动 Docker daemon（需要 root 权限）
sudo dockerd &

# 2. 启动 Qdrant 容器
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant

# 3. 验证 Qdrant 运行状态
curl http://localhost:6333/health

# 4. 设置环境变量使用 Qdrant
export VECTOR_DB=qdrant
export QDRANT_URL=http://localhost:6333
```

### 方案 4: 使用 Docker Compose

创建 `docker-compose.yml`:

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_storage:/qdrant/storage

volumes:
  qdrant_storage:
```

启动服务：
```bash
docker-compose up -d
```

## 验证 Docker 安装

```bash
# 检查 Docker 版本
docker --version

# 检查 Docker Compose 版本
docker-compose --version

# 检查 Docker daemon 状态
docker ps
```

## 故障排除

### Docker daemon 无法启动

**错误：** `Cannot connect to the Docker daemon`

**解决方案：**
1. 检查是否有 systemd 支持
2. 尝试手动启动: `sudo dockerd`
3. 检查权限: 确保用户有权限访问 `/var/run/docker.sock`
4. 使用 Chroma 本地模式作为替代方案

### Qdrant 容器无法启动

**错误：** `Cannot connect to Docker daemon`

**解决方案：**
1. 确保 Docker daemon 正在运行
2. 检查端口是否被占用: `netstat -tuln | grep 6333`
3. **推荐：使用 Qdrant 本地嵌入式模式作为替代方案（无需 Docker）**

## 推荐配置

对于开发和测试环境，推荐使用 **Qdrant 本地嵌入式模式**，因为：
- 无需 Docker 或额外服务
- 配置简单
- 高性能
- 数据持久化在本地 `./qdrant_db` 目录
- 开箱即用

对于生产环境，可以根据需求选择：
- **Qdrant 本地嵌入式模式**：单机高性能场景（推荐）
- **Qdrant 服务器模式**：分布式部署、多服务访问
- **Chroma 本地模式**：轻量级替代方案
- **Chroma 服务器模式**：易于部署和管理
