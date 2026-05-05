# 贡献指南

感谢你对 **EconometricsML** 项目的兴趣！本文档提供了如何贡献代码、报告问题和改进项目的指导。

## 行为准则

我们承诺在我们的社区中提供一个受欢迎的、有利的环境。请：

- 使用包容性语言
- 尊重不同观点和经验
- 接受建设性批评
- 关注对社区最有利的事情
- 对其他社区成员表示同情

## 如何贡献

### 1. 报告错误

错误报告通过 [GitHub Issues](https://github.com/yourusername/econometricsml/issues) 提交。

请在报告中包含：

- **描述**：清楚简洁的问题描述
- **步骤**：复现问题的详细步骤
- **预期行为**：你期望发生什么
- **实际行为**：实际发生了什么
- **环境信息**：Python版本、操作系统等
- **代码示例**：最小可复现示例

### 2. 建议改进

改进建议也通过 [Issues](https://github.com/yourusername/econometricsml/issues) 或 [Discussions](https://github.com/yourusername/econometricsml/discussions) 提交。

请说明：

- **动机**：为什么这个改进有用
- **提议的解决方案**：简述你的想法
- **替代方案**：是否有其他方式实现

### 3. 提交代码

#### 准备工作

1. Fork本仓库
2. 克隆你的fork:
   ```bash
   git clone https://github.com/yourusername/econometricsml.git
   cd econometricsml
   ```

3. 创建虚拟环境：
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

4. 安装开发依赖：
   ```bash
   pip install -e ".[dev]"
   ```

#### 开发工作流

1. **创建新分支**：
   ```bash
   git checkout -b feature/your-feature-name
   ```
   
   分支命名约定：
   - `feature/` - 新功能
   - `bugfix/` - 错误修复
   - `docs/` - 文档改进
   - `refactor/` - 代码重构

2. **编写代码**，遵循下列规范：

   - **代码风格**：使用 `black` 格式化代码
     ```bash
     black econml/
     ```
   
   - **导入排序**：使用 `isort`
     ```bash
     isort econml/
     ```
   
   - **Linting**：通过 `flake8` 检查
     ```bash
     flake8 econml/
     ```

3. **添加测试**：

   在 `tests/` 目录为新功能添加单元测试。
   
   ```python
   # tests/test_ols.py
   import pytest
   import numpy as np
   from econml.econometric_models import OLSRegression
   
   def test_ols_fit():
       X = np.random.randn(100, 3)
       y = X @ np.array([1, -0.5, 0.3]) + np.random.randn(100) * 0.1
       
       model = OLSRegression()
       model.fit(X, y)
       
       assert model.coefficients is not None
       assert model.r_squared > 0.5
   ```
   
   运行测试：
   ```bash
   pytest tests/
   ```

4. **编写文档**：

   - 在代码中添加清晰的文档字符串（docstrings）
   - 遵循NumPy文档风格
   - 如果是重大功能，在 `docs/` 中添加说明

5. **提交更改**：

   ```bash
   git add .
   git commit -m "类型: 简述改动

   更详细的说明（可选）。解释为什么做了这个改动。
   "
   ```
   
   提交信息约定：
   - `feat:` - 新功能
   - `fix:` - 错误修复
   - `docs:` - 文档
   - `style:` - 代码格式
   - `refactor:` - 重构
   - `test:` - 测试
   - `chore:` - 构建、依赖等

6. **推送并创建PR**：

   ```bash
   git push origin feature/your-feature-name
   ```
   
   然后在GitHub上创建Pull Request。

#### Pull Request 指南

PR应该：

- 有清晰的标题和描述
- 链接相关的Issue（如果存在）
- 包含对所做改动的简要说明
- 通过所有自动化测试
- 如果添加了功能，包含相应的测试
- 包含文档更新

示例PR描述：

```markdown
## 描述
修复OLS模型中的一个错误，其中...

## 相关Issue
Fixes #123

## 改动类型
- [x] 错误修复
- [ ] 新功能
- [ ] 中断式改动

## 测试
- [ ] 添加了新的测试用例
- [x] 现有测试通过

## 清单
- [x] 代码遵循项目风格
- [x] 添加了文档
- [x] 更新了changelog
```

## 代码规范

### Python 风格

遵循 [PEP 8](https://www.python.org/dev/peps/pep-0008/) 和 [Google风格指南](https://google.github.io/styleguide/pyguide.html)：

```python
"""模块级别的文档字符串"""

import numpy as np
from typing import Tuple, Optional


class MyModel:
    """
    模型的简短描述。
    
    更详细的描述。
    
    Parameters
    ----------
    param1 : int
        参数1的说明
    param2 : str, optional
        参数2的说明，默认为None
    
    Attributes
    ----------
    attribute1 : float
        属性1的说明
    
    Examples
    --------
    >>> model = MyModel(param1=10)
    >>> model.fit(X, y)
    >>> predictions = model.predict(X_test)
    """
    
    def __init__(self, param1: int, param2: Optional[str] = None):
        self.param1 = param1
        self.param2 = param2
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        拟合模型。
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            训练特征
        y : ndarray, shape (n_samples,)
            训练目标
        
        Returns
        -------
        self : object
            返回self便于链式调用
        """
        # 实现代码
        return self
```

### 命名约定

- 类名：`PascalCase` (例如 `OLSRegression`)
- 函数名：`snake_case` (例如 `calculate_metrics`)
- 常数：`UPPER_CASE` (例如 `MAX_ITERATIONS`)
- 私有方法：前缀下划线 (例如 `_internal_method`)

## 文档

### 添加示例

在 `examples/` 目录添加完整的、可运行的示例：

```python
"""
简短描述

详细描述...

运行方式: python examples/my_example.py
"""

def main():
    """主函数"""
    # 代码...
    pass

if __name__ == "__main__":
    main()
```

### 更新README

如果添加了新功能，请更新 [README.md](README.md)。

## 获取帮助

- **Questions**：在 [Discussions](https://github.com/yourusername/econometricsml/discussions) 中提问
- **Issues**：用于报告错误和建议改进
- **邮件**：your.email@example.com

## 许可证

通过贡献，你同意你的贡献将在MIT许可证下许可。

## 致谢

感谢所有贡献者使这个项目更好！

---

**最后更新**: 2024年5月
