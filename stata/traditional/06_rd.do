/*==============================================================
  文件名：06_rd.do
  模型：  断点回归设计（Regression Discontinuity Design, RDD）
  描述：  法定饮酒年龄（21岁）对死亡率的影响
  依赖：  rdrobust（ssc install rdrobust）
          rddensity（ssc install rddensity）
          esttab（ssc install estout）
  数据：  合成年龄-死亡率数据（模拟 Carpenter & Dobkin 2009）
  参考：  Carpenter, C., & Dobkin, C. (2009). The Effect of Alcohol
          Consumption on Mortality: Regression Discontinuity Evidence
          from the Minimum Drinking Age. AEJ: Applied Economics, 1(1).
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
* 1. 生成合成数据
* ---------------------------------------------------------------
* 运行变量：age - 21（以 0 为断点），范围 [-3, 3] 年
set obs 2000
gen age_centered = runiform(-3, 3)
gen above_cutoff = (age_centered >= 0)

* 真实 LATE = 8（死亡率每10万人增加8）
gen mortality = 90 + 15*age_centered - 5*age_centered^2 ///
                + 8*above_cutoff + rnormal(0, 5)

label var mortality    "全因死亡率（每10万人）"
label var age_centered "年龄距 21 岁（岁，运行变量）"
label var above_cutoff "是否达到饮酒年龄（1=≥21岁）"

* ---------------------------------------------------------------
* 2. 视觉检验：断点处跳跃
* ---------------------------------------------------------------
* 分箱散点图（Stata 内置 binscatter 或手动）
* 若安装了 binscatter：
* binscatter mortality age_centered, rd(0) line(lfit) ///
*     title("断点两侧死亡率") xtitle("年龄距 21 岁") ytitle("死亡率")

* 手动分组均值图
gen age_bin = round(age_centered * 4) / 4   // 0.25 岁为一组
preserve
    collapse (mean) mortality, by(age_bin above_cutoff)
    sort age_bin
    twoway (scatter mortality age_bin if !above_cutoff, mc(blue)) ///
           (scatter mortality age_bin if  above_cutoff, mc(red)) ///
           (lfit mortality age_bin if !above_cutoff) ///
           (lfit mortality age_bin if  above_cutoff), ///
           xline(0, lp(dash)) legend(off) ///
           title("RD：法定饮酒年龄与死亡率") ///
           xtitle("年龄距 21 岁") ytitle("死亡率（每10万）")
restore

* ---------------------------------------------------------------
* 3. 参数断点回归（本地线性，手动）
* ---------------------------------------------------------------
* 3.1 全带宽（±2 年），线性
eststo rd_lin: reg mortality c.age_centered##i.above_cutoff if abs(age_centered)<=2, robust

* 3.2 全带宽，二次项
eststo rd_quad: reg mortality c.age_centered##c.age_centered##i.above_cutoff ///
    if abs(age_centered)<=2, robust

* 3.3 窄带宽（±1 年）
eststo rd_narrow: reg mortality c.age_centered##i.above_cutoff if abs(age_centered)<=1, robust

di "LATE（本地线性，±2年）："
lincom 1.above_cutoff

esttab rd_lin rd_quad rd_narrow using "$out/table_rd.csv", ///
    replace star(* 0.1 ** 0.05 *** 0.01) b(3) se(3) ///
    keep(1.above_cutoff) ///
    title("RD：法定饮酒年龄对死亡率的影响") ///
    mtitles("线性±2年" "二次±2年" "线性±1年")

* ---------------------------------------------------------------
* 4. rdrobust（IK 最优带宽，推荐）
* ---------------------------------------------------------------
rdrobust mortality age_centered, c(0) kernel(triangular)
* 结果：LATE 及其稳健置信区间

* ---------------------------------------------------------------
* 5. 密度连续性检验（McCrary 2008 / rddensity）
* ---------------------------------------------------------------
rddensity age_centered, c(0)
* p > 0.05：无法拒绝运行变量在断点处连续（支持 RD 有效性）

* ---------------------------------------------------------------
* 6. 协变量平衡检验（安慰剂）
* ---------------------------------------------------------------
* 以前定基线协变量为结果变量，断点处不应有跳跃
* 此处用 noise 模拟基线特征
gen baseline_health = 50 + 2*age_centered + rnormal(0, 3)
rdrobust baseline_health age_centered, c(0) kernel(triangular)
di "协变量安慰剂：p 值应 > 0.1"

di "06_rd.do 运行完毕"
