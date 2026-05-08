# notebooks — 论文级示例 Notebook

每个 Notebook 对应一篇真实的经济学/金融学论文复现，遵循**六段式**结构。

## traditional/ — 传统计量方法（T01–T11，全部完成）

| 文件 | 方法 | 参考论文 |
|------|------|---------|
| T01_OLS_Mincer工资方程.ipynb | OLS 完整推断 | Mincer (1974) |
| T01_OLS_最小可运行示例.ipynb | OLS 最简示例 | — |
| T02_Logit_企业出口决策.ipynb | Logit / Probit | Roberts & Tybout (1997) |
| T03_IV_制度质量与经济发展.ipynb | 工具变量 / 2SLS | Acemoglu et al. (2001) |
| T04_Panel_企业生产率.ipynb | 面板固定效应 | Olley & Pakes (1996) |
| T05_DiD_最低工资政策.ipynb | 双重差分 | Card & Krueger (1994) |
| T06_RD_法定饮酒年龄与死亡率.ipynb | 回归断点 | Carpenter & Dobkin (2009) |
| T07_PSM_政府补贴研发效应.ipynb | 倾向得分匹配 | Heckman et al. (1997) |
| T08_EventStudy_股权质押公告效应.ipynb | 事件研究 | Fama et al. (1969) |
| T09_GARCH_A股波动率.ipynb | GARCH(1,1) | Bollerslev (1986) |
| T10_VAR_货币政策传导.ipynb | VAR / 脉冲响应 | Sims (1980) |
| T11_SyntheticControl_政策效应.ipynb | 合成控制法 | Abadie & Gardeazabal (2003) |

## 六段式规范

每个 Notebook 必须包含（顺序不变）：

```
## 0. 论文信息
## 1. 研究设计与识别策略
## 2. 数学理论与模型
## 3. 数据加载与预处理
## 4. 模型估计
## 5. 结果解读与稳健性检验
## 6. 可视化
```

## 其他目录（预留）

| 目录 | 状态 |
|------|------|
| ml/ | 机器学习方法（待开发） |
| dl/ | 深度学习（待开发） |
| rl/ | 强化学习（待开发） |
| llm/ | LLM 应用（待开发） |
