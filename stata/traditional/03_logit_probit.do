/*==============================================================
  文件名：03_logit_probit.do
  模型：  二元离散选择模型（Logit / Probit）
  描述：  企业出口决策——规模、生产率、FDI 的影响
  依赖：  marginscontplot（可选，ssc install marginscontplot）
          esttab / estout（ssc install estout）
  数据：  合成面板数据（模拟 Roberts & Tybout 1997 AER 设计）
  参考：  Roberts, M. J., & Tybout, J. R. (1997). The Decision to
          Export in Colombia: An Empirical Model of Entry with Sunk
          Costs. American Economic Review, 87(4), 545–564.
  作者：  （留空，由使用者填写）
  更新：  2025-05
==============================================================*/

* ---------------------------------------------------------------
* 0. 全局设置
* ---------------------------------------------------------------
global root  "C:/Users/yourname/empirlab"
global data  "$root/data/sample"
global out   "$root/output"

clear all
set more off
set seed 42

* ---------------------------------------------------------------
* 1. 生成合成数据（替换为真实数据时删除此节）
* ---------------------------------------------------------------
* DGP：export = 1 if  -2 + 0.8*lnsize + 1.2*lnprod + 0.5*fdi + eps > 0
set obs 500
gen lnsize = rnormal(5, 1)
gen lnprod = lnsize * 0.4 + rnormal(0, 0.8)
gen fdi    = (runiform() < 0.3)
gen xb     = -2 + 0.8*lnsize + 1.2*lnprod + 0.5*fdi
gen export = (xb + rnormal(0, 1) > 0)
label var export  "是否出口（1=出口）"
label var lnsize  "企业规模（对数销售额）"
label var lnprod  "劳动生产率（对数）"
label var fdi     "是否外资参股（1=是）"

* ---------------------------------------------------------------
* 2. 描述性统计
* ---------------------------------------------------------------
sum export lnsize lnprod fdi
tab export

* ---------------------------------------------------------------
* 3. Logit 估计
* ---------------------------------------------------------------
eststo logit1: logit export lnsize lnprod fdi, robust
estadd scalar pseudo_r2 = e(r2_p)

* Logit + 二次项（非线性检验）
eststo logit2: logit export c.lnsize##c.lnsize lnprod fdi, robust
estadd scalar pseudo_r2 = e(r2_p)

* ---------------------------------------------------------------
* 4. Probit 估计（对比）
* ---------------------------------------------------------------
eststo probit1: probit export lnsize lnprod fdi, robust
estadd scalar pseudo_r2 = e(r2_p)

* ---------------------------------------------------------------
* 5. 平均边际效应（AME）
* ---------------------------------------------------------------
* Logit AME
quietly logit export lnsize lnprod fdi, robust
margins, dydx(*)
estimates store ame_logit

* Probit AME
quietly probit export lnsize lnprod fdi, robust
margins, dydx(*)
estimates store ame_probit

* ---------------------------------------------------------------
* 6. 输出回归表
* ---------------------------------------------------------------
esttab logit1 logit2 probit1 using "$out/table_logit.csv", ///
    replace star(* 0.1 ** 0.05 *** 0.01) ///
    b(3) se(3) ///
    scalars("pseudo_r2 Pseudo R2" "N N") ///
    title("Logit/Probit：企业出口决策") ///
    mtitles("Logit" "Logit(quad)" "Probit")

* ---------------------------------------------------------------
* 7. 预测概率与分类准确率
* ---------------------------------------------------------------
quietly logit export lnsize lnprod fdi, robust
predict p_hat, pr

gen y_hat = (p_hat >= 0.5)
gen correct = (y_hat == export)
sum correct
di "分类准确率（阈值=0.5）：" round(r(mean)*100, 0.01) "%"

* Sensitivity / Specificity
tab export y_hat

* ---------------------------------------------------------------
* 8. 拟合优度诊断
* ---------------------------------------------------------------
* Hosmer-Lemeshow 检验
quietly logit export lnsize lnprod fdi
estat gof, group(10) table

* ---------------------------------------------------------------
* 9. 稳健性检验
* ---------------------------------------------------------------
* 9.1 剔除极端值
quietly sum lnsize, detail
gen flag_trim = (lnsize < r(p5) | lnsize > r(p95))
eststo logit_trim: logit export lnsize lnprod fdi if !flag_trim, robust

* 9.2 加入行业固定效应（需要行业变量）
* 若有 industry 变量：
* eststo logit_fe: logit export lnsize lnprod fdi i.industry, robust

esttab logit1 logit_trim using "$out/table_logit_robust.csv", ///
    replace star(* 0.1 ** 0.05 *** 0.01) b(3) se(3) ///
    title("稳健性：Logit 企业出口决策") ///
    mtitles("基准" "缩尾样本")

di "03_logit_probit.do 运行完毕"
