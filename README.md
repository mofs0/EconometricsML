# EconometricsML

> **Reusable empirical research code templates for economics · finance · business**
> Python package: `empirlab` | 面向经济 / 金融 / 商科的实证研究代码库

---

## 💡 为什么用 Python 做实证，而不是 Stata？

用 Stata 做实证，有一个痛点几乎人人都经历过：**回归结果不显著，就得重找指标、重下数据、重新清洗，反复折腾，一篇论文磨几个月。**

这不是个人问题，是 Stata 工作方式决定的：每换一个变量，前面的手工操作几乎得重来一遍，大量时间花在体力劳动上，研究本身反而成了次要的。

**Python 解决的正是这个问题。**

**① 换指标只要改一行，不用重新下数据**

```python
import akshare as ak
df = ak.stock_financial_abstract(symbol="000001")  # 直接拉财务数据
```

换指标？改一行，重新跑，几秒出结果。Stata 用户可能需要半天。

**② 批量测试多个模型，自动对比，不靠手动记录**

```python
for var in ['lev', 'roa', 'size', 'age', 'tobinq']:
    res = smf.ols(f'y ~ {var} + controls', data=df).fit()
    print(f'{var}: coef={res.params[var]:.4f}, p={res.pvalues[var]:.4f}')
```

**③ 结果不显著？用机器学习自动筛变量，不靠碰运气**

```python
from sklearn.linear_model import LassoCV
lasso = LassoCV().fit(X, y)
important_vars = X.columns[lasso.coef_ != 0]  # 自动找出真正重要的指标
```

**④ 全流程一条管道，改一处自动更新，不怕漏跑某步**

**⑤ 完全免费**，不需要 Stata 授权费

> **一句话：Stata 用户的时间，很多花在"不显著→重找数据"的死循环里。Python 把这个循环变成几行代码，把时间还给真正的研究。**

---

## 📦 仓库里有什么

### 一、传统计量方法（T01–T11，全部完成）

论文级 Notebook，每个示例对应一篇经典实证论文的研究设计：

| 编号 | 方法 | 应用场景 |
|------|------|----------|
| T01 | OLS / Mincer 工资方程 | 教育回报率估计 |
| T02 | Logit 回归 | 企业出口决策 |
| T03 | IV / 2SLS | 制度质量与经济发展（殖民史作工具变量）|
| T04 | 面板固定效应 | 企业全要素生产率 |
| T05 | 双重差分 DID | 最低工资政策效果评估 |
| T06 | 回归断点 RD | 法定饮酒年龄与死亡率 |
| T07 | 倾向得分匹配 PSM | 政府补贴对研发的因果效应 |
| T08 | 事件研究法 | 股权质押公告的市场反应 |
| T09 | GARCH | A 股波动率建模 |
| T10 | VAR / 脉冲响应 | 货币政策传导机制 |
| T11 | 合成控制法 | 政策干预效应评估 |

### 二、论文写作全流程（academic/）

从选题到投稿的完整指南，7 个专题 Notebook：

| 编号 | 主题 | 核心内容 |
|------|------|----------|
| A01 | 文献检索与高效阅读 | 检索平台、关键词策略、WWH 阅读框架、创新点挖掘 |
| A02 | 选题与开题报告 | 选题风险管理、五段式开题结构、答辩 Q&A |
| A03 | 期刊选择与投稿 | SCI/SSCI/EI + CSSCI/北大核心/普刊 全体系，投稿流程 |
| A04 | SCI 小论文写作 | 各章节写法、三必备图表、审稿意见回复模板 |
| A05 | 毕业大论文 | 章节结构、三年写作时间线、查重与答辩准备 |
| A06 | 学术裁缝写作方法 | 四种拼接策略 + 基准模型插件式写法 |
| A07 | 数据获取服务 | A 股/宏观/债券/基金数据范围介绍，联系：1795837192@qq.com |

---

## 🚀 快速上手（5 步）

### 第一步：安装 Python

打开 [python.org/downloads](https://www.python.org/downloads/)，下载安装，**务必勾选 `Add Python to PATH`**。

验证：在命令提示符输入 `python --version`，能看到版本号就成功。

### 第二步：下载代码

**新手推荐**：点 GitHub 页面右上角绿色 `Code` 按钮 → `Download ZIP` → 解压到本地。

**有 Git 的**：
```bash
git clone https://github.com/mofs0/EconometricsML.git
cd EconometricsML
```

### 第三步：安装依赖

```bash
pip install -r requirements.txt
pip install -e .
```

> 网速慢加镜像：`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

### 第四步：打开 Jupyter

```bash
jupyter notebook
```

浏览器自动打开，进入 `notebooks/traditional/`，点开任意 `.ipynb`，按 `Shift+Enter` 逐格运行。

### 第五步：验证环境

```bash
python empirlab/traditional/ols.py
```

看到回归结果表格，说明一切正常 🎉

---

## 🐍 完全零基础？先看 Python 入门教程

`python_tutorial/` 文件夹里有 9 个专为本项目设计的 Python 入门 Notebook（P00–P08），从零开始到能独立跑实证回归，全程用经济/金融场景举例，没有编程基础也能学。

---

## 📁 项目结构

```text
EconometricsML/
├── academic/                    # 论文写作全流程（A01–A07）
│   ├── A01_文献检索与高效阅读.ipynb
│   ├── A02_选题与开题报告.ipynb
│   ├── A03_期刊选择与投稿.ipynb
│   ├── A04_SCI小论文写作全指南.ipynb
│   ├── A05_毕业大论文.ipynb
│   ├── A06_学术裁缝写作方法.ipynb
│   └── A07_数据获取服务.ipynb
│
├── notebooks/
│   └── traditional/             # T01–T11 传统计量（全部完成）
│
├── empirlab/                    # Python 核心方法库
│   ├── traditional/             # OLS / IV / Panel / DiD / RD / PSM / EventStudy / GARCH / VAR / SC
│   └── utils/                   # 数据读写 / 可视化 / 预处理 / 评估指标
│
├── python_tutorial/             # Python 入门教程（P00–P08，零基础可学）
├── stata/                       # Stata 对应代码（traditional/ + ml/）
└── rerun_all_notebooks.ps1      # Windows 一键重跑 + 推送脚本
```

---

## 💻 使用示例

### OLS 回归

```python
from empirlab.traditional.ols import OLS, mincer_data

df = mincer_data(n=500)
model = OLS(robust=True).fit(df[["educ", "exper", "exper2"]], df["ln_wage"])
print(model.summary_table().round(4))
```

---

## 📋 Notebook 六段式规范

每个 `.ipynb` 统一结构：

```
## 0. 论文信息
## 1. 研究设计与识别策略
## 2. 数学理论与模型
## 3. 数据加载与预处理
## 4. 模型估计
## 5. 结果解读与稳健性检验
## 6. 可视化
```

---

## ❓ 常见问题

**Q：`ModuleNotFoundError: No module named 'xxx'`**  
A：运行 `pip install xxx`（把 xxx 换成报错里的包名）。

**Q：`pip install` 速度慢或超时**  
A：`pip install xxx -i https://pypi.tuna.tsinghua.edu.cn/simple`

**Q：Jupyter 运行报错 `Kernel died`**  
A：内存不足或包版本冲突，Kernel → Restart 重启再试。

**Q：不会用 Git，能直接下 ZIP 吗？**  
A：可以，见上方"第二步"方式 A。

---

## ⚖️ 法律声明

本仓库内容仅供**学术交流与学习使用**。

- ✅ 允许：学习参考、课堂教学、非盈利科研
- ❌ 禁止：倒卖、转售、打包为付费课程对外销售

违反上述规定的商业行为，将依法追究法律责任。本声明受中华人民共和国相关法律法规保护。

---

本项目采用 [MIT License](LICENSE) 开源协议。
