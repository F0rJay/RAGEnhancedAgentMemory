# PostgreSQL 配置指南

PostgreSQL 密码需要你自己设置。以下是不同场景下的设置方法。

## 🚀 快速开始

### 方式一：使用 Docker（推荐，最简单）

```bash
# 启动 PostgreSQL 容器并设置密码
docker run --name postgres-memory \
  -e POSTGRES_PASSWORD=mysecretpassword \
  -e POSTGRES_DB=agent_memory \
  -p 5432:5432 \
  -d postgres:15

# 然后在 .env 文件中设置
POSTGRES_PASSWORD=mysecretpassword
```

**说明**：
- `POSTGRES_PASSWORD` 就是你设置的密码（示例中是 `mysecretpassword`）
- 你可以改成任何你想要的密码，只要在 `.env` 文件中保持一致即可

### 方式二：已安装 PostgreSQL（Linux/Mac）

#### 1. 如果忘记了密码，可以重置：

```bash
# 停止 PostgreSQL 服务
sudo systemctl stop postgresql

# 编辑配置文件，允许本地免密登录
sudo nano /etc/postgresql/*/main/pg_hba.conf
# 找到这一行并修改为：
# local   all             all                                     trust

# 重启服务
sudo systemctl start postgresql

# 使用 postgres 用户登录（无需密码）
sudo -u postgres psql

# 在 psql 中重置密码
ALTER USER postgres PASSWORD 'your_new_password';

# 退出
\q

# 恢复 pg_hba.conf 配置
sudo nano /etc/postgresql/*/main/pg_hba.conf
# 改回：
# local   all             all                                     md5

# 重启服务
sudo systemctl restart postgresql
```

#### 2. 查看当前配置：

```bash
# 连接到 PostgreSQL
sudo -u postgres psql

# 查看数据库列表
\l

# 查看用户
\du

# 退出
\q
```

### 方式三：Windows 上安装 PostgreSQL

1. **下载安装**：从 https://www.postgresql.org/download/windows/ 下载安装程序
2. **安装时设置密码**：安装过程中会提示你设置 postgres 用户的密码
3. **记住这个密码**：这就是你在 `.env` 文件中要填写的密码

### 方式四：使用云服务（AWS RDS、Azure 等）

云服务提供商会在创建数据库实例时让你设置密码：

- **AWS RDS**：在创建数据库实例时设置主密码（Master Password）
- **Azure Database**：在创建时设置管理员密码
- **Google Cloud SQL**：在创建实例时设置根密码

## 📝 在项目中配置

### 1. 创建数据库（如果不存在）

```bash
# 使用 Docker
docker exec -it postgres-memory psql -U postgres

# 或在已安装的 PostgreSQL 中
sudo -u postgres psql

# 创建数据库
CREATE DATABASE agent_memory;

# 退出
\q
```

### 2. 在 .env 文件中设置

```bash
# 复制示例文件
cp env.example .env

# 编辑 .env 文件，填写密码
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=agent_memory
POSTGRES_USER=postgres
POSTGRES_PASSWORD=你设置的密码  # 这里填写你刚才设置的密码
```

## 🔍 验证连接

### 测试连接是否成功

```bash
# 使用 Docker
docker exec -it postgres-memory psql -U postgres -d agent_memory

# 或使用 psql 命令行
psql -h localhost -U postgres -d agent_memory

# 如果连接成功，你会看到：
# agent_memory=#
```

### 使用 Python 测试

```python
import psycopg2

try:
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="agent_memory",
        user="postgres",
        password="your_password"  # 填写你的密码
    )
    print("连接成功！")
    conn.close()
except Exception as e:
    print(f"连接失败: {e}")
```

## 🛠️ 常见场景

### 场景 1：全新安装（使用 Docker）

```bash
# 1. 启动 PostgreSQL
docker run --name postgres-memory \
  -e POSTGRES_PASSWORD=agent123 \
  -e POSTGRES_DB=agent_memory \
  -p 5432:5432 \
  -d postgres:15

# 2. 在 .env 中设置
POSTGRES_PASSWORD=agent123
```

### 场景 2：使用已有 PostgreSQL

```bash
# 1. 连接到 PostgreSQL
sudo -u postgres psql

# 2. 创建新用户（可选）
CREATE USER agent_user WITH PASSWORD 'agent123';

# 3. 创建数据库
CREATE DATABASE agent_memory OWNER agent_user;

# 4. 授权
GRANT ALL PRIVILEGES ON DATABASE agent_memory TO agent_user;

# 5. 在 .env 中设置
POSTGRES_USER=agent_user
POSTGRES_PASSWORD=agent123
```

### 场景 3：忘记密码

```bash
# 使用 Docker（最简单）
docker stop postgres-memory
docker rm postgres-memory
# 重新创建容器并设置新密码
docker run --name postgres-memory \
  -e POSTGRES_PASSWORD=new_password \
  -e POSTGRES_DB=agent_memory \
  -p 5432:5432 \
  -d postgres:15

# 在 .env 中更新
POSTGRES_PASSWORD=new_password
```

## ⚠️ 注意事项

1. **密码安全**：
   - 使用强密码（包含大小写字母、数字、特殊字符）
   - 不要使用默认密码
   - 不要将密码提交到 Git

2. **备份**：
   - 记住你的密码
   - 考虑使用密码管理器

3. **开发环境 vs 生产环境**：
   - 开发环境可以使用简单密码
   - 生产环境必须使用强密码

## 🔗 相关资源

- [PostgreSQL 官方文档](https://www.postgresql.org/docs/)
- [Docker PostgreSQL 镜像](https://hub.docker.com/_/postgres)
- [psycopg2 文档](https://www.psycopg.org/docs/)

## ❓ 常见问题

### Q: 我找不到密码，怎么办？

A: 
- 如果是 Docker：重新创建容器并设置新密码
- 如果是已安装的：使用重置密码的方法
- 如果是云服务：在控制台重置密码

### Q: 可以不使用 PostgreSQL 吗？

A: 目前项目依赖 PostgreSQL 存储会话状态和元数据。如果不需要检查点功能，可以暂时跳过配置，但某些功能可能不可用。

### Q: 如何更改密码？

A:
- Docker：删除容器并重新创建
- 已安装：使用 `ALTER USER` 命令
- 云服务：在控制台重置
