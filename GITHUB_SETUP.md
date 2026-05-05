# GitHub 上传指南

本项目已在本地初始化了Git仓库。按照下面的步骤将其上传到GitHub。

## 步骤1: 在GitHub上创建新仓库

1. 访问 https://github.com/new
2. 填写信息：
   - **Repository name**: `econometricsml`
   - **Description**: `A comprehensive library for econometric and financial machine learning models`
   - **Public** (公开) 或 **Private** (私有) 根据你的需要选择
   - ❌ **不要** 勾选 "Initialize this repository with README" 或其他选项

3. 点击 "Create repository"

## 步骤2: 添加Remote并推送代码

创建仓库后，GitHub会显示推送指令。按照以下操作：

```bash
# 1. 进入项目目录
cd d:\Git\EconometricsML

# 2. 添加远程仓库（替换 YOUR_USERNAME）
git remote add origin https://github.com/YOUR_USERNAME/econometricsml.git

# 3. 重命名主分支为main（GitHub新仓库的默认分支）
git branch -M main

# 4. 推送代码到GitHub
git push -u origin main
```

## 步骤3: 验证推送成功

在你的GitHub仓库页面应该能看到所有的代码文件。

## 后续操作

### 更新仓库信息

编辑GitHub仓库设置：

1. **Topics**（标签）：添加以下标签便于搜索
   - `econometrics`
   - `machine-learning`
   - `time-series`
   - `finance`
   - `deep-learning`
   - `reinforcement-learning`
   - `python`

2. **Homepage**（主页）：（可选）设置指向文档的链接

3. **Enable Discussions**：启用讨论功能便于用户提问

### 配置GitHub Pages（用于在线文档）

1. 进入仓库设置 → Pages
2. 选择 "Source" → "main branch"
3. 选择文件夹为 "/docs"

## 本地开发工作流

```bash
# 1. 创建新分支
git checkout -b feature/new-feature

# 2. 做改动，然后提交
git add .
git commit -m "feat: 添加新特性"

# 3. 推送到GitHub
git push origin feature/new-feature

# 4. 在GitHub上创建Pull Request，merge后：
git checkout main
git pull origin main

# 5. 删除本地分支
git branch -d feature/new-feature
```

## 常用Git命令

```bash
# 查看提交历史
git log --oneline

# 查看当前状态
git status

# 查看远程仓库
git remote -v

# 更新本地代码
git pull origin main

# 创建标签（发布版本时）
git tag -a v0.1.0 -m "Version 0.1.0"
git push origin v0.1.0
```

## 配置GitHub Actions（可选）

创建 `.github/workflows/tests.yml` 文件以自动运行测试：

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', '3.11']
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
    
    - name: Run tests
      run: |
        pytest tests/
```

## 遇到问题？

### 常见错误1: 远程仓库拒绝推送

```
error: failed to push some refs to...
```

**解决方案**:
```bash
git pull origin main --rebase
git push origin main
```

### 常见错误2: 认证失败

使用Personal Access Token而不是密码：

1. GitHub Settings → Developer settings → Personal access tokens
2. 生成新token（至少需要 `repo` 权限）
3. 使用token作为密码：
   ```bash
   git push origin main
   # 用户名: YOUR_USERNAME
   # 密码: 你的token
   ```

或使用SSH（更安全）：

```bash
# 生成SSH密钥（如果没有的话）
ssh-keygen -t ed25519 -C "your.email@example.com"

# 添加到SSH agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# 在GitHub上添加公钥：Settings → SSH and GPG keys → New SSH key
cat ~/.ssh/id_ed25519.pub

# 使用SSH URL
git remote set-url origin git@github.com:YOUR_USERNAME/econometricsml.git
```

## 发布版本

当准备发布新版本时：

```bash
# 1. 更新版本号
# 编辑 econml/__init__.py 和 pyproject.toml 中的版本号

# 2. 提交版本更新
git add econml/__init__.py pyproject.toml
git commit -m "chore: bump version to v0.2.0"

# 3. 创建标签
git tag -a v0.2.0 -m "Release version 0.2.0"

# 4. 推送
git push origin main
git push origin v0.2.0
```

## 更新README中的GitHub链接

将README.md和其他文档中的以下占位符替换为你的实际信息：

- `yourusername` → 你的GitHub用户名
- `your.email@example.com` → 你的邮箱

使用全局查找替换：

```bash
# Linux/Mac
sed -i 's/yourusername/YOUR_USERNAME/g' README.md
sed -i 's/your.email@example.com/YOUR_EMAIL/g' README.md

# Windows PowerShell
(Get-Content README.md) -replace 'yourusername', 'YOUR_USERNAME' | Set-Content README.md
(Get-Content README.md) -replace 'your.email@example.com', 'YOUR_EMAIL' | Set-Content README.md
```

---

祝你成功发布项目！如有问题，参考 [GitHub官方文档](https://docs.github.com/)。
