# Stata 联动工作流

这份文档只讲 Stata 路线，适合主力回归放在 Stata、但希望借助 Python 做清洗和结果整理的场景。

## 1. 分工建议

- Python 负责：数据清洗、合并、特征工程、可视化、机器学习和结果整理
- Stata 负责：传统计量回归、面板回归、稳健性检验和论文表格输出

## 2. 数据往返

先在 Python 中准备或清洗数据，再导出给 Stata：

```python
import pandas as pd

df = pd.read_csv('data/your_file.csv')
df.to_stata('data/your_file.dta', write_index=False)
```

在 Stata 中读取：

```stata
use "data/your_file.dta", clear
describe
reg y x1 x2
xtset firmid year
xtreg y x1 x2, fe
```

如果 Stata 已经处理完数据或结果，你也可以读回 Python：

```python
import pandas as pd

result_df = pd.read_stata('data/your_file.dta')
```

## 3. 适合的使用场景

如果你不熟 Python，可以先把 Python 当作“数据处理器”，只做下面几件事：

1. 读取 CSV / Excel / Stata 数据。
2. 做变量清洗和格式转换。
3. 导出 `.dta` 给 Stata。
4. 把 Stata 的结果再读回 Python 做画图或汇总。

## 4. 常见提醒

- Stata 里注意变量名不要太长，也不要含有奇怪符号。
- 导出前先确认编码和缺失值处理方式。
- 如果是面板数据，最好保留 `firmid` 和 `year` 这类索引变量。
