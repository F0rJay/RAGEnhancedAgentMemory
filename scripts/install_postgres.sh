#!/bin/bash
# PostgreSQL 安装脚本（不使用 Docker）

echo "=== PostgreSQL 安装（非 Docker 方式）==="
echo ""

# 检测系统类型
if command -v apt-get &> /dev/null; then
    # Ubuntu/Debian
    echo "检测到 Ubuntu/Debian 系统"
    echo "正在安装 PostgreSQL..."
    
    sudo apt-get update
    sudo apt-get install -y postgresql postgresql-contrib
    
    # 启动服务
    sudo service postgresql start
    
    # 设置密码
    PASSWORD=${POSTGRES_PASSWORD:-rag_memory_2024}
    echo "设置 postgres 用户密码..."
    sudo -u postgres psql -c "ALTER USER postgres PASSWORD '$PASSWORD';"
    
    # 创建数据库
    sudo -u postgres createdb agent_memory 2>/dev/null || echo "数据库已存在"
    
    echo ""
    echo "✅ PostgreSQL 安装完成！"
    echo ""
    echo "连接信息："
    echo "  Host: localhost"
    echo "  Port: 5432"
    echo "  Database: agent_memory"
    echo "  User: postgres"
    echo "  Password: $PASSWORD"
    echo ""
    echo "请在 .env 文件中设置："
    echo "  POSTGRES_PASSWORD=$PASSWORD"
    
elif command -v yum &> /dev/null; then
    # CentOS/RHEL
    echo "检测到 CentOS/RHEL 系统"
    echo "请手动安装 PostgreSQL: sudo yum install postgresql-server postgresql-contrib"
    exit 1
else
    echo "❌ 不支持的 Linux 发行版"
    exit 1
fi
