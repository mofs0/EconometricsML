/*==============================================================
  文件名：04_panel_fe_re.do
  模型：  面板固定效应 / 随机效应 + Hausman 检验
  描述：  企业层面生产率决定因素——Cobb-Douglas 框架
  依赖：  xttest0、hausman（Stata 内置）
          esttab（ssc install estout）
  数据：  合成企业面板数据（模拟鲁晓东 & 连玉君 2012 设计）
  参考：  鲁晓东, 连玉君 (2012). 中国工业企业的全要素生产率估计.
          《经济学(季刊)》11(2), 541–558.
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
* 1. 生成合成面板数据
* ---------------------------------------------------------------
* 100 家企业 × 8 年
local N_firm 100
local T_year 8

set obs `= `N_firm' * `T_year''
gen firm_id = ceil(_n / `T_year')
gen year    = 2010 + mod(_n - 1, `T_year')

* 企业固定效应
by firm_id, sort: gen u_i = rnormal(0, 0.5) if _n == 1
by firm_id: replace u_i = u_i[1]

* 解释变量
gen lnK   = rnormal(8, 1)   // 资本对数
gen lnL   = rnormal(6, 0.8) // 劳动对数
gen age   = runiform(1, 30)  // 企业年龄
gen lnage = ln(age)

* 结果变量（Cobb-Douglas + 固定效应）
gen lnY = 0.4*lnK + 0.5*lnL - 0.02*lnage + u_i + rnormal(0, 0.3)

label var lnY   "产出对数（ln GDP）"
label var lnK   "资本对数"
label var lnL   "劳动对数"
label var lnage "企业年龄对数"

xtset firm_id year

* ---------------------------------------------------------------
* 2. 描述性统计
* ---------------------------------------------------------------
xtsum lnY lnK lnL lnage

* ---------------------------------------------------------------
* 3. 混合 OLS（POLS，忽略个体异质性）
* ---------------------------------------------------------------
eststo pols: reg lnY lnK lnL lnage, cluster(firm_id)

* ---------------------------------------------------------------
* 4. 固定效应（FE / Within）
* ---------------------------------------------------------------
eststo fe: xtreg lnY lnK lnL lnage, fe vce(cluster firm_id)
estimates store fe_est

* F 检验：固定效应联合显著性
test

* ---------------------------------------------------------------
* 5. 随机效应（RE / GLS）
* ---------------------------------------------------------------
eststo re: xtreg lnY lnK lnL lnage, re vce(robust)
estimates store re_est

* Breusch-Pagan LM 检验：是否需要随机效应
xttest0

* ---------------------------------------------------------------
* 6. Hausman 检验（FE vs RE）
* ---------------------------------------------------------------
hausman fe_est re_est, sigmamore
* p < 0.05：拒绝 H0，使用 FE；p >= 0.05：无法拒绝，RE 有效

* ---------------------------------------------------------------
* 7. 时间固定效应（双向固定效应）
* ---------------------------------------------------------------
eststo twfe: xtreg lnY lnK lnL lnage i.year, fe vce(cluster firm_id)

* ---------------------------------------------------------------
* 8. 结果输出
* ---------------------------------------------------------------
esttab pols fe re twfe using "$out/table_panel.csv", ///
    replace star(* 0.1 ** 0.05 *** 0.01) ///
    b(3) se(3) ///
    stats(N r2 r2_within r2_between, fmt(0 3 3 3)) ///
    title("面板估计：企业生产率（Cobb-Douglas）") ///
    mtitles("POLS" "FE" "RE" "双向FE")

* ---------------------------------------------------------------
* 9. 稳健性检验
* ---------------------------------------------------------------
* 9.1 剔除首尾各 1% 极端值
foreach v of varlist lnY lnK lnL {
    quietly sum `v', detail
    replace `v' = . if `v' < r(p1) | `v' > r(p99)
}
eststo fe_trim: xtreg lnY lnK lnL lnage, fe vce(cluster firm_id)

esttab fe fe_trim using "$out/table_panel_robust.csv", ///
    replace star(* 0.1 ** 0.05 *** 0.01) b(3) se(3) ///
    title("稳健性：面板固定效应") ///
    mtitles("基准 FE" "缩尾 FE")

di "04_panel_fe_re.do 运行完毕"
