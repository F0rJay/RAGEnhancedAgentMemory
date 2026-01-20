# GitHub SSH 配置说明

## SSH 公钥

已为你生成 SSH 密钥对。请将以下公钥添加到你的 GitHub 账号：

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMGcyckzVYxAlLyXp3XgKWxE7Gi9fVGtFvDVCDdBs0W1 github-autodl
```

## 添加步骤

1. 访问 GitHub: https://github.com/settings/keys
2. 点击 "New SSH key"
3. 标题填写: `autodl-server`（或任意名称）
4. Key 类型选择: `Authentication Key`
5. Key 内容粘贴上面的公钥
6. 点击 "Add SSH key"

## 验证配置

添加完成后，运行以下命令验证：

```bash
ssh -T git@github.com
```

如果看到 "Hi F0rJay! You've successfully authenticated..." 说明配置成功。

## 推送代码

配置成功后，运行：

```bash
cd /root/autodl-tmp/RAGEnhancedAgentMemory
git push origin main
```
