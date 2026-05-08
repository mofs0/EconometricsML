# EmpirLab

**面向定量研究的工程化代码仓库。**

聚焦**经济学 · 金融学 · 商科**实证研究，提供可复用的计量方法代码模板；
`academic` 分支另含论文写作全流程指南与数据获取服务。

> Python 包名：`empirlab` · Git 仓库：`EconometricsML`

---

## 🙋 第一次来？先看这里！

**不懂 Python？没用过 GitHub？完全没关系，按下面的步骤一步一步来就行。**

本项目专为经济/金融/商科的同学设计，**不需要编程基础**也能跑起来。如果你是完全的新手，建议先去看本仓库的 Python 入门教学（见下方 `python_tutorial/` 文件夹），再回来使用这里的代码模板。

---

## 💡 为什么用 Python 做实证，而不是 Stata？

很多经济/金融专业的同学做实证都在用 Stata，但有一个痛点几乎人人都经历过：**回归结果不显著，就得重新找指标、换数据，反复折腾**。这背后的原因，很大程度上是 Stata 的工作方式决定的。

### Stata 的典型困境

用 Stata 做实证，流程通常是这样的：

1. 手动下载数据（Wind、CSMAR、手工收集……）
2. 导入 Stata，清洗、合并
3. 跑回归，结果不显著
4. **回到第1步，重找指标，重新下载，重新清洗……**

每换一个指标，前面的工作几乎得重来一遍。一篇论文来回折腾几个月，大量时间花在"找数据→不显著→再找数据"的死循环里，**研究本身反而成了次要的**。

### Python 怎么解决这个问题

Python 的核心优势不在于"更先进"，而在于**让你把精力放在研究设计上，而不是重复的体力劳动上**。

**① 数据获取自动化，换指标只需改一行代码**

用 `akshare`、`tushare`、`pandas-datareader` 等库，直接从网上拉数据，不用手动下载 Excel：

```python
import akshare as ak
df = ak.stock_financial_abstract(symbol="000001")  # 一行拿到财务数据
```

换一个指标？改一行，重新跑，几秒钟出结果。Stata 用户可能需要半天。

**② 批量尝试多个模型，自动对比结果**

Python 可以一次性跑几十个回归变体，自动输出对比表格：

```python
# 一次性测试 5 个不同的核心解释变量
for var in ['lev', 'roa', 'size', 'age', 'tobinq']:
    result = smf.ols(f'y ~ {var} + controls', data=df).fit()
    print(f'{var}: coef={result.params[var]:.4f}, p={result.pvalues[var]:.4f}')
```

Stata 做同样的事情，需要手动改代码、逐个跑、逐个记录。

**③ 机器学习辅助变量筛选，不靠"碰运气"**

结果不显著时，Python 可以用 LASSO、Random Forest 等方法，**从几十个候选变量里自动找出真正重要的指标**，而不是靠直觉和运气：

```python
from sklearn.linear_model import LassoCV
lasso = LassoCV().fit(X, y)
important_vars = X.columns[lasso.coef_ != 0]  # 自动筛出显著变量
```

**④ 全流程可复现，改一处全部更新**

Python 脚本从数据读取到最终表格是一条完整的流水线，改了数据或方法，**一键重跑，所有结果自动更新**。Stata 一般需要手动重跑多个 do 文件，容易漏掉某一步。

**⑤ 免费，没有授权费**

Stata 需要付费授权，学校提供的版本有时还有功能限制。Python 完全免费，所有库也免费。

---

### 一句话总结

> **Stata 用户的时间，很多花在"结果不显著→重找数据"的循环里。Python 让你把这个循环变成几行代码，把时间还给真正的研究。**

---

## 🚀 从零开始：环境配置教程

### 第一步：安装 Python

1. 打开 [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. 点击黄色大按钮 **Download Python 3.x.x**（选最新版本即可）
3. 运行安装包，**务必勾选底部的 `Add Python to PATH`**，然后点 Install Now

> ✅ 验证安装成功：打开命令提示符（Windows 按 `Win+R` 输入 `cmd` 回车），输入 `python --version`，能看到版本号就成功了。

---

### 第二步：下载本项目代码

**方式 A（推荐新手）：直接下载 ZIP 压缩包**

1. 进入本仓库主页（你现在看的这个页面）
2. 点击右上角绿色按钮 **`Code`**
3. 点击 **`Download ZIP`**
4. 解压到你喜欢的文件夹，比如 `D:\EconometricsML`

**方式 B：用 Git 克隆（如果你已经装了 Git）**

```bash
git clone https://github.com/你的用户名/EconometricsML.git
cd EconometricsML
```

---

### 第三步：安装依赖包

打开命令提示符，进入你解压/克隆的文件夹（把路径换成你自己的）：

```bash
# 进入项目文件夹（把路径换成你自己的）
cd D:\EconometricsML

# 安装所有依赖包（等待几分钟，国内网络慢可以换镜像，见下方提示）
pip install -r requirements.txt

# 安装本项目自身的包
pip install -e .
```

> 💡 **网速慢？用国内镜像加速：**
> ```bash
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

---

### 第四步：打开 Jupyter Notebook 运行示例

本项目的示例都是 `.ipynb` 格式（Jupyter Notebook），可以像 Word 一样边看说明边运行代码。

```bash
# 在项目目录下运行
jupyter notebook
```

浏览器会自动打开，进入 `notebooks/traditional/` 文件夹，点开任意一个 `.ipynb` 文件，按 **`Shift + Enter`** 逐格运行代码。

> 💡 **如果提示 `jupyter` 命令找不到：**
> ```bash
> pip install notebook
> ```

---

### 第五步：运行一个最简单的示例验证一切正常

```bash
# 在命令提示符中运行（在项目目录下）
python empirlab/traditional/ols.py
```

看到一张回归结果表格，就说明环境配置完全成功 🎉

---

## 🐍 Python 入门教学（专为本项目定制）

**第一次接触 Python？** 我们在 `python_tutorial/` 文件夹中准备了一套专门为本项目定制的 Python 入门教程，内容紧贴经济/金融实证场景，学完就能直接用本项目的代码。

```text
python_tutorial/
├── P00_写在前面_如何使用本教程.ipynb       # 教程导读，必看
├── P01_Python基础_变量与数据类型.ipynb     # 数字、字符串、列表、字典
├── P02_Python基础_条件与循环.ipynb         # if/for/while
├── P03_Python基础_函数与模块.ipynb         # def、import 用法
├── P04_数据处理_NumPy数组.ipynb            # 矩阵运算基础
├── P05_数据处理_Pandas表格操作.ipynb       # 读取/清洗数据，最常用！
├── P06_数据可视化_Matplotlib画图.ipynb     # 折线图、散点图、直方图
├── P07_统计基础_描述性统计与假设检验.ipynb  # 均值、方差、t检验
└── P08_实战演练_用Python复现一篇实证论文.ipynb  # 综合案例
```

> 📌 **学习路径建议：**
> - 完全零基础：P00 → P01 → P02 → P03 → P05 → 直接看 notebooks/
> - 有一点基础（会 Excel）：P04 → P05 → P06 → 直接看 notebooks/
> - 只想跑通代码：看完 P03 就够了

---

## 1. 仓库定位

- **`main` 分支**：可复用、可审计、可扩展的实证研究代码模板（传统计量 + ML + DL + RL + LLM）
- **`academic` 分支**：论文写作从选题到投稿的完整指南（A01–A07），含中英文期刊写作、学术裁缝方法、数据获取服务

---

## 2. 项目结构

```text
EconometricsML/
├── python_tutorial/             # 🆕 Python 入门教程（专为本项目定制，新手必看）
│   ├── P00_写在前面_如何使用本教程.ipynb
│   ├── P01_Python基础_变量与数据类型.ipynb
│   ├── P02_Python基础_条件与循环.ipynb
│   ├── P03_Python基础_函数与模块.ipynb
│   ├── P04_数据处理_NumPy数组.ipynb
│   ├── P05_数据处理_Pandas表格操作.ipynb
│   ├── P06_数据可视化_Matplotlib画图.ipynb
│   ├── P07_统计基础_描述性统计与假设检验.ipynb
│   └── P08_实战演练_用Python复现一篇实证论文.ipynb
│
├── empirlab/                    # Python 包（核心方法库）
│   ├── traditional/             # 经典计量方法
│   │   ├── ols.py               ✅ 完整推断（SE/t/p/F/HC1/adj-R²）
│   │   ├── iv.py                ✅ 2SLS / 工具变量
│   │   ├── panel.py             ✅ 固定效应 / 随机效应
│   │   ├── did.py               ✅ 双重差分
│   │   ├── rd.py                ✅ 回归断点
│   │   ├── psm.py               ✅ 倾向得分匹配
│   │   ├── event_study.py       ✅ 事件研究 / CAR
│   │   ├── garch.py             ✅ GARCH 波动率建模
│   │   ├── var.py               ✅ VAR / 脉冲响应函数
│   │   └── synthetic_control.py ✅ 合成控制法
│   ├── ml/                      # 机器学习方法
│   │   ├── double_ml.py         ✅ DoubleML（Neyman 正交）
│   │   ├── causal_forest.py     ✅ 因果森林
│   │   ├── regularized.py       ✅ Lasso / Ridge / ElasticNet
│   │   └── tree_models.py       ✅
│   ├── dl/                      # 深度学习（PyTorch）
│   │   ├── mlp_regressor.py     ✅
│   │   └── mlp.py               ✅
│   ├── rl/                      # 强化学习（Gymnasium）
│   │   └── q_learning_agent.py  ✅
│   ├── llm/                     # LLM 调用封装
│   │   └── prompt_client.py     ✅
│   └── utils/                   ✅ metrics / visualization / preprocessing / data_io
│
├── notebooks/
│   └── traditional/             # T01–T11 论文级示例 Notebook（全部完成）
│       ├── T01_OLS_Mincer工资方程.ipynb        ✅
│       ├── T01_OLS_最小可运行示例.ipynb         ✅
│       ├── T02_Logit_企业出口决策.ipynb         ✅
│       ├── T03_IV_制度质量与经济发展.ipynb      ✅
│       ├── T04_Panel_企业生产率.ipynb           ✅
│       ├── T05_DiD_最低工资政策.ipynb           ✅
│       ├── T06_RD_法定饮酒年龄与死亡率.ipynb    ✅
│       ├── T07_PSM_政府补贴研发效应.ipynb       ✅
│       ├── T08_EventStudy_股权质押公告效应.ipynb ✅
│       ├── T09_GARCH_A股波动率.ipynb            ✅
│       ├── T10_VAR_货币政策传导.ipynb           ✅
│       └── T11_SyntheticControl_政策效应.ipynb  ✅
│
├── academic/                    # 论文写作系列（academic 分支）
│   ├── A01_文献检索与高效阅读.ipynb
│   ├── A02_选题与开题报告.ipynb
│   ├── A03_期刊选择与投稿.ipynb
│   ├── A04_SCI小论文写作全指南.ipynb
│   ├── A05_毕业大论文.ipynb
│   ├── A06_学术裁缝写作方法.ipynb
│   └── A07_数据获取服务.ipynb
│
├── stata/
│   ├── traditional/
│   │   ├── 01_ols.do             ✅ 完整 Mincer 示例
│   │   └── 02_iv_2sls.do
│   └── ml/
│       └── 01_double_ml.do       ✅
│
└── rerun_all_notebooks.ps1       # Windows 一键重跑 + 推送脚本
```

---

## 3. 快速开始

> 💡 **如果你是第一次使用，请先看上方"从零开始：环境配置教程"，再来看这里。**

```bash
pip install -r requirements.txt
pip install -e .

# 直接运行最小示例（无需额外数据，可以验证环境是否正常）
python empirlab/traditional/ols.py
python empirlab/ml/double_ml.py
python empirlab/dl/mlp_regressor.py
python empirlab/rl/q_learning_agent.py
python empirlab/llm/prompt_client.py
```

---

## 4. 使用示例

### OLS（完整推断）

```python
from empirlab.traditional.ols import OLS, mincer_data

df = mincer_data(n=500)
model = OLS(robust=True).fit(df[["educ", "exper", "exper2"]], df["ln_wage"])
print(model.summary_table().round(4))
# coef / se(HC1) / t / p_value / ci_lower / ci_upper
```

### DoubleML

```python
import numpy as np
from empirlab.ml import DoubleML

rng = np.random.default_rng(42)
n, X = 400, rng.standard_normal((400, 5))
d = 0.4 * X[:, 0] + rng.standard_normal(n) * 0.3
y = 1.2 * d + 0.5 * X[:, 0] + rng.standard_normal(n) * 0.5
model = DoubleML().fit(X, y, d)
print(model.summary())
```

---

## 5. Notebook 规范（六段式）

每个 `.ipynb` 必须包含以下六段标题，顺序不变：

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

## 6. academic 分支：论文写作指南

切换分支后可访问 `academic/` 文件夹，包含 7 个专题 Notebook：

| 编号 | 主题 | 核心内容 |
|------|------|----------|
| A01 | 文献检索与高效阅读 | 平台选择、关键词策略、WWH 阅读框架、创新点挖掘 |
| A02 | 选题与开题报告 | 选题风险管理、五段式开题结构、答辩 Q&A |
| A03 | 期刊选择与投稿 | SCI/SSCI/EI + **CSSCI/北大核心/普刊** 全体系，投稿流程 |
| A04 | SCI 小论文写作 | 各章节写法、三必备图表、审稿意见逐条回复模板 |
| A05 | 毕业大论文 | 大论文结构、三年写作时间线、查重与答辩准备 |
| A06 | 学术裁缝写作方法 | 四种拼接策略（情境迁移/机制叠加/测度改进/异质性扩展） |
| A07 | 数据获取服务 | A 股/宏观/债券/基金数据范围介绍，联系：1795837192@qq.com |

```bash
# 切换到 academic 分支（需要先安装 Git）
git checkout academic
```

---

## 7. 常见问题 FAQ

**Q：运行时提示 `ModuleNotFoundError: No module named 'xxx'`**
A：说明某个包没装上，运行 `pip install xxx` 即可（把 xxx 换成报错里的包名）。

**Q：`pip install` 速度很慢或超时**
A：换清华镜像源：`pip install xxx -i https://pypi.tuna.tsinghua.edu.cn/simple`

**Q：打开 Jupyter Notebook 后，运行代码报错 `Kernel died`**
A：多半是内存不足或包版本冲突，重启内核（菜单栏 Kernel → Restart）再试一次。

**Q：我不会用 Git，能不能不用 Git？**
A：完全可以。直接下载 ZIP 压缩包即可，参考上方"第二步：下载本项目代码"方式 A。

**Q：academic 分支的内容在哪里？**
A：需要用 Git 切换分支（`git checkout academic`）。如果你只下载了 ZIP，建议重新下载时在 GitHub 页面切换到 `academic` 分支再下载。

---

## 8. 代码规范

详见 [`.github/copilot-instructions.md`](.github/copilot-instructions.md)。核心约束：

- 模型类只放 `empirlab/` 对应子包，不在 notebook 里重写
- 每个文件：独立类 + 完整 docstring + `if __name__ == "__main__"` 最小示例
- 深度学习统一 PyTorch，强化学习统一 Gymnasium
- `utils/` 统一管理工具函数

---

## 9. Stata 对应

| Python | Stata | 状态 |
|--------|-------|------|
| `traditional/ols.py` | `stata/traditional/01_ols.do` | ✅ 完整 |
| `traditional/iv.py` | `stata/traditional/02_iv_2sls.do` | 骨架 |
| `ml/double_ml.py` | `stata/ml/01_double_ml.do` | ✅ 完整 |

---

## 10. 其他学科（预留，未开发）

架构已预留以下学科的扩展入口：

- **生物统计 / 医学**：生存分析（Cox PH、KM）、随机对照试验、荟萃分析
- **社会学 / 政策**：调查方法、多层次模型
- **管理学**：结构方程模型（SEM）
- **教育学**：教育效果评估（DID/IV）、学业成就影响因素、教育经济学实证
- **计算机科学**：实证软件工程、用户行为建模、在线实验（A/B Test）、推荐系统评估

如需扩展，新增对应子包目录即可，无需改动现有代码。

---

## 11. 法律声明

本仓库内容（代码、Notebook、文档）仅供**学术交流与学习使用**。

- ✅ 允许：学习参考、课堂教学、非盈利科研使用
- ❌ 禁止：以任何形式倒卖、转售或将本仓库内容用于商业牟利
- ❌ 禁止：将本仓库内容打包为付费课程、付费资料对外销售

**违反上述规定的商业行为，将依法追究法律责任。**

> 本声明受中华人民共和国相关法律法规保护。

---

## 12. 许可证


本项目采用 [MIT License](LICENSE) 开源协议。
