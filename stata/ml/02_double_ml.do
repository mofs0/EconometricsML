/* 02_double_ml.do
   Double ML Stata 模板（占位）。
*/

global root "C:/path/to/empirlab"  // 用户请修改为本地路径
global data "$root/data/sample"
global out "$root/output"

cd "$root"
clear all
set more off

* 载入数据（示例）
use "$data/firm_panel.dta", clear

* 这里放置双重机器学习实现或调用 pdslasso 等步骤
display "请在此处实现 Double ML 的 Stata 步骤"
