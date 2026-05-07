# Stata + Python 联动工作流

本文档用于说明如何在 Stata 中调用 Python、以及如何把 Python 清洗后的数据导回 Stata 做回归分析。

## 1. 在 Stata 中指定 Python

```stata
python set exec "C:/path/to/python.exe"
```

## 2. 在 Python 中进行数据清洗

```python
from empirlab.utils.data_io import read_csv

df = read_csv("data/sample.csv")
```

## 3. 结果导回 Stata

```stata
import delimited "output/cleaned.csv", clear
```
