/*==============================================================
  文件名：09_garch.do
  模型：  GARCH(p,q) —— 条件异方差与波动率预测
  描述：  A 股市场收益率波动率建模（日度数据）
  依赖：  arch（Stata 内置）
          esttab（ssc install estout）
  数据：  合成日度收益率（模拟方意 & 荣雪 2019 设计）
  参考：  方意, 荣雪 (2019). 中国股票市场波动率的测量与预测.
          《管理世界》35(7).
          Bollerslev, T. (1986). Generalized Autoregressive
          Conditional Heteroskedasticity. Journal of Econometrics.
  作者：  （留空，由使用者填写）
  更新：  2025-05
==============================================================*/

* ---------------------------------------------------------------
* 0. 全局设置
* ---------------------------------------------------------------
global root  "C:/Users/yourname/empirlab"
global out   "$root/output"

clear all
set more off
set seed 2025

* ---------------------------------------------------------------
* 1. 生成合成日度收益率（GARCH(1,1) DGP）
* ---------------------------------------------------------------
* 参数：omega=0.00002, alpha=0.10, beta=0.85
local T 1000
local omega 0.00002
local alpha 0.10
local beta  0.85

set obs `T'
gen t = _n
gen date = td(01jan2020) + _n
format date %td

* 模拟 GARCH(1,1)
gen h     = 0.0001 in 1
gen eps   = rnormal(0, 1)
gen ret   = 0 in 1
replace ret = sqrt(h) * eps in 1

forvalues i = 2/`T' {
    quietly replace h = `omega' + `alpha' * ret[_n-1]^2 + `beta' * h[_n-1] in `i'
    quietly replace eps = rnormal(0, 1) in `i'
    quietly replace ret = sqrt(h) * eps in `i'
}
replace ret = ret * 100   // 转换为百分比

tsset t

label var ret  "日度收益率（%）"
label var h    "条件方差（真实）"

* ---------------------------------------------------------------
* 2. 描述性统计
* ---------------------------------------------------------------
sum ret
sktest ret        // 偏度-峰度检验
swilk ret         // Shapiro-Wilk 正态性检验

* ARCH-LM 检验（是否存在条件异方差）
reg ret L.ret
estat archlm, lags(1 5 10)

* ---------------------------------------------------------------
* 3. GARCH(1,1) 估计（正态分布）
* ---------------------------------------------------------------
eststo garch11: arch ret, arch(1) garch(1) vce(robust)

* ---------------------------------------------------------------
* 4. 对比：GARCH(1,1)-t 分布
* ---------------------------------------------------------------
eststo garch11t: arch ret, arch(1) garch(1) distribution(t) vce(robust)

* ---------------------------------------------------------------
* 5. EGARCH(1,1)（非对称效应）
* ---------------------------------------------------------------
eststo egarch: arch ret, earch(1) egarch(1) vce(robust)

* ---------------------------------------------------------------
* 6. GJR-GARCH（门限效应）
* ---------------------------------------------------------------
* arch 命令中使用 tarch 选项
eststo gjr: arch ret, arch(1) tarch(1) garch(1) vce(robust)

* ---------------------------------------------------------------
* 7. 信息准则比较
* ---------------------------------------------------------------
estimates stats garch11 garch11t egarch gjr

* ---------------------------------------------------------------
* 8. 波动率预测（条件方差序列）
* ---------------------------------------------------------------
quietly arch ret, arch(1) garch(1) vce(robust)
predict h_hat, variance
label var h_hat "条件方差（GARCH 预测）"

twoway (line h_hat t, lc(blue) lw(thin)) ///
       (line h t, lc(red) lp(dash) lw(thin)), ///
       legend(label(1 "GARCH(1,1) 预测") label(2 "真实条件方差")) ///
       title("波动率：GARCH(1,1) vs 真实") ///
       xtitle("交易日") ytitle("条件方差")

* 5 日向前预测
arch ret, arch(1) garch(1)
forecast create garch_fcast, replace
forecast estimates garch11
forecast solve, prefix(f_) begin(e(N)+1) periods(5)

* ---------------------------------------------------------------
* 9. 稳健性检验
* ---------------------------------------------------------------
* 9.1 次样本稳健性（前后各一半）
local half = int(`T'/2)
eststo garch_h1: arch ret in 1/`half', arch(1) garch(1) vce(robust)
eststo garch_h2: arch ret in `= `half'+1'/`T', arch(1) garch(1) vce(robust)

esttab garch11 garch11t gjr garch_h1 garch_h2 using "$out/table_garch.csv", ///
    replace star(* 0.1 ** 0.05 *** 0.01) b(5) se(5) ///
    title("GARCH 模型比较：A 股波动率") ///
    mtitles("GARCH(1,1)" "GARCH-t" "GJR-GARCH" "前半段" "后半段")

di "09_garch.do 运行完毕"
