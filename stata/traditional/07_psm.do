/*==============================================================
  文件名：07_psm.do
  模型：  倾向得分匹配（Propensity Score Matching, PSM）
  描述：  政府补贴对企业研发投入的因果效应
  依赖：  psmatch2（ssc install psmatch2）
          pstest（psmatch2 自带）
          esttab（ssc install estout）
  数据：  合成企业截面数据（模拟余明桂等 2010 设计）
  参考：  余明桂, 回雅甫, 潘红波 (2010). 政治联系、寻租与地方政府
          财政补贴有效性. 《经济研究》(3), 65–77.
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
* 1. 生成合成数据
* ---------------------------------------------------------------
set obs 1000
gen lnsize  = rnormal(8, 1.5)
gen lnage   = ln(abs(rnormal(10, 5)) + 1)
gen leverage= rbeta(2, 5)
gen roa     = rnormal(0.05, 0.03)
gen state   = (runiform() < 0.3)

* 选择方程：大企业、国有、ROA 高更可能获补贴
gen xb_treat = -1.5 + 0.4*lnsize + 0.5*state + 3*roa + rnormal(0, 0.8)
gen subsidy  = (xb_treat > 0)

* 结果变量：R&D / 总资产，ATT = 0.02
gen rd_ratio = 0.03 + 0.01*lnsize - 0.01*leverage + 0.02*roa ///
               + 0.02*subsidy + rnormal(0, 0.015)
replace rd_ratio = max(rd_ratio, 0)

label var rd_ratio "R&D 投入强度（R&D/资产）"
label var subsidy  "是否获得政府补贴"
label var lnsize   "企业规模（对数）"
label var lnage    "企业年龄（对数）"
label var leverage "资产负债率"
label var roa      "资产收益率"
label var state    "国有企业"

* ---------------------------------------------------------------
* 2. 朴素 OLS（不匹配，存在选择偏差）
* ---------------------------------------------------------------
eststo ols: reg rd_ratio subsidy lnsize lnage leverage roa state, robust
di "OLS 估计（有偏）"

* ---------------------------------------------------------------
* 3. 倾向得分估计（Logit）
* ---------------------------------------------------------------
logit subsidy lnsize lnage leverage roa state
predict pscore, pr

sum pscore if subsidy==1
sum pscore if subsidy==0

* 共同支撑检验
twoway (kdensity pscore if subsidy==1, lc(red)) ///
       (kdensity pscore if subsidy==0, lc(blue)), ///
       legend(label(1 "处置组") label(2 "对照组")) ///
       title("倾向得分分布") xtitle("倾向得分")

* ---------------------------------------------------------------
* 4. 最近邻匹配（1:1，有放回，caliper=0.05）
* ---------------------------------------------------------------
psmatch2 subsidy lnsize lnage leverage roa state, ///
    outcome(rd_ratio) n(1) caliper(0.05) ate ate noreplace common

* 匹配后协变量平衡检验
pstest lnsize lnage leverage roa state, both graph

* ---------------------------------------------------------------
* 5. 不同匹配算法的 ATT
* ---------------------------------------------------------------
* 5.1 最近邻 1:2
psmatch2 subsidy lnsize lnage leverage roa state, ///
    outcome(rd_ratio) n(2) caliper(0.05) common

* 5.2 核匹配（Epanechnikov 核，带宽 0.06）
psmatch2 subsidy lnsize lnage leverage roa state, ///
    outcome(rd_ratio) kernel kerneltype(epan) bwidth(0.06) common

* ---------------------------------------------------------------
* 6. ATT 汇总输出
* ---------------------------------------------------------------
di "PSM ATT 汇总（见 psmatch2 输出）"
* 若需汇总表，可手动记录各 ATT 值

* ---------------------------------------------------------------
* 7. 稳健性检验
* ---------------------------------------------------------------
* 7.1 Rosenbaum Bounds（隐藏偏差检验，需 rbounds）
* rbounds rd_ratio, gamma(1.0 1.5 2.0 2.5) nboot(500)

* 7.2 安慰剂结果变量（用与补贴无关的变量）
gen placebo_var = rnormal(0, 1)
psmatch2 subsidy lnsize lnage leverage roa state, ///
    outcome(placebo_var) n(1) caliper(0.05) common
di "安慰剂 ATT 应接近 0"

di "07_psm.do 运行完毕"
