/*==============================================================
  文件名：10_var.do
  模型：  向量自回归（VAR）+ IRF + FEVD + Granger 因果
  描述：  货币政策传导机制：货币供给、物价、产出的动态关系
  依赖：  irf（Stata 内置 VAR 后命令）
          esttab（ssc install estout）
  数据：  合成月度宏观时间序列（模拟陈六傅 & 刘厚俊 2008 设计）
  参考：  陈六傅, 刘厚俊 (2008). 人民币汇率的价格传递效应——基于
          VAR 模型的实证分析. 《经济研究》(7), 52–64.
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
set seed 42

* ---------------------------------------------------------------
* 1. 生成合成月度宏观数据（T=120 个月，2014m1–2023m12）
* ---------------------------------------------------------------
set obs 120
gen t = _n
gen date = ym(2014, 1) + _n - 1
format date %tm
tsset date

* DGP：简化 VAR(2) 结构
* lnm2:  货币供给量（对数）
* lncpi: 价格水平（对数）
* lnrgdp: 实际产出（对数）
* r:     利率

gen lnm2   = 0 in 1/2
gen lncpi  = 0 in 1/2
gen lnrgdp = 0 in 1/2
gen r      = 0.04 in 1/2

forvalues i = 3/120 {
    quietly replace lnm2   = 0.05 + 0.7*lnm2[_n-1]   + 0.1*lnm2[_n-2]   ///
                             - 0.1*r[_n-1]   + rnormal(0, 0.01) in `i'
    quietly replace lncpi  = 0.002 + 0.5*lncpi[_n-1]  + 0.2*lncpi[_n-2]  ///
                             + 0.15*lnm2[_n-1] + rnormal(0, 0.008) in `i'
    quietly replace lnrgdp = 0.008 + 0.6*lnrgdp[_n-1] + 0.15*lnrgdp[_n-2] ///
                             + 0.1*lnm2[_n-1] - 0.05*r[_n-1] + rnormal(0, 0.012) in `i'
    quietly replace r      = 0.005 + 0.8*r[_n-1]      + 0.1*lncpi[_n-1]  ///
                             - 0.05*lnrgdp[_n-1] + rnormal(0, 0.002) in `i'
}

label var lnm2   "货币供给 M2（对数）"
label var lncpi  "价格水平 CPI（对数）"
label var lnrgdp "实际 GDP（对数）"
label var r      "利率"

* ---------------------------------------------------------------
* 2. 单位根检验
* ---------------------------------------------------------------
foreach v of varlist lnm2 lncpi lnrgdp r {
    dfuller `v', lags(4) trend
    pperron `v', lags(4) trend
}

* 若为 I(1)，取一阶差分
gen dlnm2   = D.lnm2
gen dlncpi  = D.lncpi
gen dlnrgdp = D.lnrgdp
gen dr      = D.r

* ---------------------------------------------------------------
* 3. 最优滞后阶选择
* ---------------------------------------------------------------
varsoc dlnm2 dlncpi dlnrgdp dr, maxlag(6)
* 选择 AIC / BIC 最小对应的滞后阶

* ---------------------------------------------------------------
* 4. VAR 估计
* ---------------------------------------------------------------
var dlnm2 dlncpi dlnrgdp dr, lags(1/2)
estimates store var2

* 残差诊断
varlmar, mlag(4)   // LM 自相关检验
varstable           // 特征根稳定性
varnorm             // 残差正态性

* ---------------------------------------------------------------
* 5. Granger 因果检验
* ---------------------------------------------------------------
vargranger
* 货币供给→CPI 因果？货币供给→产出因果？

* ---------------------------------------------------------------
* 6. 脉冲响应函数（IRF）
* ---------------------------------------------------------------
irf create var2_irf, set(irf_var2) replace step(12) order(dlnm2 dr dlncpi dlnrgdp)

* 货币供给冲击对 CPI 的响应
irf graph oirf, impulse(dlnm2) response(dlncpi) ///
    title("货币供给冲击对 CPI 的正交化脉冲响应") ///
    xtitle("月") ytitle("响应幅度")

* 利率冲击对产出的响应
irf graph oirf, impulse(dr) response(dlnrgdp) ///
    title("利率冲击对实际 GDP 的正交化脉冲响应") ///
    xtitle("月") ytitle("响应幅度")

* 汇总 4 个变量对货币供给冲击的响应
irf graph oirf, impulse(dlnm2) response(dlnm2 dlncpi dlnrgdp dr) ///
    title("货币供给冲击的 OIRF（12 期）")

* ---------------------------------------------------------------
* 7. 预测误差方差分解（FEVD）
* ---------------------------------------------------------------
irf graph fevd, impulse(dlnm2) response(dlnrgdp) ///
    title("FEVD：货币供给对产出预测方差的贡献") ///
    xtitle("月")

* 输出数值表
irf table fevd, impulse(dlnm2) response(dlncpi dlnrgdp)

* ---------------------------------------------------------------
* 8. 结构 VAR（SVAR，短期约束）——可选
* ---------------------------------------------------------------
* 递归识别：Cholesky 分解（顺序：M2 → r → CPI → GDP）
* matrix A = (1,0,0,0 \ .,1,0,0 \ .,.,1,0 \ .,.,.,1)
* matrix B = I(4)
* svar dlnm2 dr dlncpi dlnrgdp, aeq(A) beq(B)

* ---------------------------------------------------------------
* 9. 稳健性：不同变量排序
* ---------------------------------------------------------------
irf create var2_alt, set(irf_var2_alt) replace step(12) ///
    order(dr dlnm2 dlncpi dlnrgdp)

irf graph oirf, set(irf_var2_alt) impulse(dlnm2) response(dlncpi) ///
    title("货币供给→CPI（替换排序稳健性）")

di "10_var.do 运行完毕"
