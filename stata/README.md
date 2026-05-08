# stata — Stata 对照实现

与 `empirlab/` 和 `notebooks/` 一一对应的 Stata `.do` 文件，方便需要 Stata 环境的研究者直接使用。

## traditional/ — 传统计量方法

| 文件 | 方法 | 对应 Python Notebook |
|------|------|---------------------|
| 01_ols.do | OLS（Mincer 工资方程） | T01 ✅ 完整 |
| 02_iv_2sls.do | 工具变量 / 2SLS | T03 ✅ |
| 03_logit_probit.do | Logit / Probit | T02 ✅ |
| 04_panel_fe_re.do | 面板固定效应 / 随机效应 | T04 ✅ |
| 05_did.do | 双重差分（DID） | T05 ✅ |
| 06_rd.do | 回归断点（RD） | T06 ✅ |
| 07_psm.do | 倾向得分匹配（PSM） | T07 ✅ |
| 08_event_study.do | 事件研究 | T08 ✅ |
| 09_garch.do | GARCH 波动率 | T09 ✅ |
| 10_var.do | VAR / 脉冲响应 | T10 ✅ |
| 11_synthetic_control.do | 合成控制法 | T11 ✅ |

## ml/ — 机器学习（Stata 近似实现）

| 文件 | 方法 | 说明 |
|------|------|------|
| 01_double_ml.do | Double ML | 使用 Stata 17+ 内置 `lassoregress`；含 2-折交叉拟合 |

## 使用说明

1. 打开对应 `.do` 文件，修改顶部的 `global root` 路径为本地路径
2. 所有 `.do` 文件使用**合成数据**，无需下载外部数据集即可直接运行
3. 需要安装的外部命令已在文件头注释中标注（`ssc install xxx`）

## 版本要求

- Stata 16+（部分 ML 方法需要 Stata 17+ 内置 lasso）
- 推荐安装：`ivreg2`、`estout`、`