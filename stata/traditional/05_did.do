/*==============================================================
  文件名：05_did.do
  模型：  双重差分法（Difference-in-Differences, DiD）
  描述：  最低工资政策对青年就业的影响
  依赖：  esttab（ssc install estout）
          coefplot（ssc install coefplot，用于平行趋势图）
  数据：  合成城市-年度面板（模拟 Fang & Lin 2015 设计）
  参考：  Fang, T., & Lin, C. (2015). Minimum wages and employment
          in China. IZA Journal of Labor Policy, 4(1), 1–30.
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
* 1. 生成合成城市面板
* ---------------------------------------------------------------
* 60 个城市 × 10 年（2005–2014），政策 2010 年起实施
local N_city 60
local T_year 10
local treat_yr 2010

set obs `= `N_city' * `T_year''
gen city_id = ceil(_n / `T_year')
gen year    = 2004 + mod(_n - 1, `T_year') + 1   // 2005..2014

* 处置组：城市 ID <= 30
gen treat = (city_id <= 30)
gen post  = (year >= `treat_yr')
gen did   = treat * post

* 城市固定效应
by city_id, sort: gen alpha_i = rnormal(0, 0.5) if _n==1
by city_id: replace alpha_i = alpha_i[1]

* ATT = -0.05（政策使就业率下降 5 个百分点）
gen emp_rate = 0.72 + 0.02*(year-2010) + alpha_i ///
               - 0.05*did + rnormal(0, 0.04)
gen lngdp    = rnormal(10, 1)   // 控制变量：GDP

label var emp_rate "青年就业率"
label var treat    "处置组（1=政策城市）"
label var post     "政策后（1=2010年及以后）"
label var did      "DID 交互项"
label var lngdp    "人均 GDP（对数）"

xtset city_id year

* ---------------------------------------------------------------
* 2. 平行趋势检验（事件研究型）
* ---------------------------------------------------------------
* 以 2009 年为基准期，生成逐年交互项
forvalues yr = 2005/2014 {
    gen treat_`yr' = treat * (year == `yr')
}
* 基准期：2009 年（omit）
reg emp_rate treat_2005 treat_2006 treat_2007 treat_2008 ///
    treat_2010 treat_2011 treat_2012 treat_2013 treat_2014 ///
    i.city_id i.year, cluster(city_id)
* 图示：coefplot（需安装）
* coefplot, keep(treat_200? treat_201?) vertical yline(0) ///
*           xline(5.5, lp(dash)) title("平行趋势检验")

* ---------------------------------------------------------------
* 3. 基准 DiD 估计
* ---------------------------------------------------------------
* 3.1 无控制变量，含双向 FE
eststo did_base: xtreg emp_rate did i.year, fe vce(cluster city_id)

* 3.2 加入控制变量
eststo did_ctrl: xtreg emp_rate did lngdp i.year, fe vce(cluster city_id)

* 3.3 OLS + 稳健 SE（对比）
eststo did_ols: reg emp_rate did lngdp i.city_id i.year, vce(cluster city_id)

di "ATT 估计（基准）："
lincom did

* ---------------------------------------------------------------
* 4. 结果输出
* ---------------------------------------------------------------
esttab did_base did_ctrl did_ols using "$out/table_did.csv", ///
    replace star(* 0.1 ** 0.05 *** 0.01) ///
    b(4) se(4) ///
    keep(did lngdp) ///
    stats(N r2_within, fmt(0 3)) ///
    title("DiD：最低工资政策对青年就业的影响") ///
    mtitles("无控制变量" "含控制变量" "OLS")

* ---------------------------------------------------------------
* 5. 稳健性检验
* ---------------------------------------------------------------
* 5.1 安慰剂检验：虚假干预期（2008 年）
gen post_placebo = (year >= 2008)
gen did_placebo  = treat * post_placebo
eststo placebo: xtreg emp_rate did_placebo lngdp i.year if year < `treat_yr', ///
    fe vce(cluster city_id)

* 5.2 排除干预年（2010 年）
eststo excl2010: xtreg emp_rate did lngdp i.year if year != 2010, ///
    fe vce(cluster city_id)

esttab did_ctrl placebo excl2010 using "$out/table_did_robust.csv", ///
    replace star(* 0.1 ** 0.05 *** 0.01) b(4) se(4) ///
    keep(did did_placebo) ///
    title("稳健性：DiD 最低工资") ///
    mtitles("基准" "安慰剂(2008)" "排除干预年")

di "05_did.do 运行完毕"
