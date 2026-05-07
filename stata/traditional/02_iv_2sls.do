/*==============================================================
  文件名：02_iv_2sls.do
  模型：  工具变量法 / 两阶段最小二乘（IV / 2SLS）
  描述：  制度质量对人均收入的影响——定居者死亡率作为工具变量
  依赖：  ivreg2（ssc install ivreg2）
          ranktest（ssc install ranktest）
          xtoverid（ssc install xtoverid，可选）
          esttab（ssc install estout）
  数据：  合成截面数据（模拟 Acemoglu, Johnson & Robinson 2001 AER）
  参考：  Acemoglu, D., Johnson, S., & Robinson, J. A. (2001).
          The Colonial Origins of Comparative Development: An
          Empirical Investigation. AER, 91(5), 1369–1401.
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
* 1. 生成合成截面数据（N=60 个国家）
* ---------------------------------------------------------------
* DGP：
*   logmort → instit（第一阶段，IV 有效）
*   instit  → logpgdp（第二阶段，真实效应=0.9）
set obs 60

gen logmort  = rnormal(4.5, 1.2)         // 殖民者死亡率（对数），工具变量
gen instit   = 6 - 0.8*logmort + rnormal(0, 0.8)  // 制度质量（内生）
gen lat_abst = runiform(0, 1)             // 地理控制变量（纬度）
gen africa   = (runiform() < 0.25)        // 非洲虚拟变量
gen logpgdp  = 2 + 0.9*instit + 0.3*lat_abst - 0.2*africa + rnormal(0, 0.5)

label var logpgdp  "人均 GDP（对数，结果变量）"
label var instit   "制度质量指数（内生变量）"
label var logmort  "殖民者死亡率（对数，工具变量）"
label var lat_abst "纬度（地理控制）"
label var africa   "非洲国家虚拟变量"

* ---------------------------------------------------------------
* 2. 内生性诊断
* ---------------------------------------------------------------
* 2.1 朴素 OLS（内生性导致估计偏误）
eststo ols: reg logpgdp instit lat_abst africa, robust

* 2.2 约化型（Reduced Form）：工具变量对结果的直接效应
eststo rf: reg logpgdp logmort lat_abst africa, robust

* 2.3 第一阶段（First Stage）：工具变量对内生变量的效应
eststo fs: reg instit logmort lat_abst africa, robust
test logmort
local F_first = r(F)
di "第一阶段 F 统计量：" round(`F_first', 0.01) " （>10 为强工具变量）"

* ---------------------------------------------------------------
* 3. 2SLS 估计（ivreg2，推荐，报告第一阶段 F 统计量）
* ---------------------------------------------------------------
ivreg2 logpgdp lat_abst africa (instit = logmort), robust first
estimates store iv_2sls
estadd scalar first_F = e(widstat)      // Cragg-Donald / Kleibergen-Paap F

* ---------------------------------------------------------------
* 4. 弱工具变量检验
* ---------------------------------------------------------------
* Kleibergen-Paap rk Wald F 统计量（ivreg2 自动报告）
* 若 F > 10（Stock-Yogo 10% 临界值），工具变量为强工具变量

* ---------------------------------------------------------------
* 5. 过度识别检验（仅当有多个工具变量时）
* ---------------------------------------------------------------
* 添加第二个工具变量（欧洲殖民史长度，模拟）
gen euro_col = rnormal(0, 1) - 0.3*logmort       // 与 logmort 弱相关
ivreg2 logpgdp lat_abst africa (instit = logmort euro_col), robust
* estat overid 或 ivreg2 自动报告 Sargan-Hansen J 统计量（p>0.1 为通过）

* ---------------------------------------------------------------
* 6. 内生性检验（Hausman / Wu-Hausman）
* ---------------------------------------------------------------
ivreg2 logpgdp lat_abst africa (instit = logmort), robust endog(instit)
* C 统计量（p<0.05 说明 instit 确实内生，需要 IV）

* ---------------------------------------------------------------
* 7. 结果输出
* ---------------------------------------------------------------
esttab ols fs rf iv_2sls using "$out/table_iv.csv", ///
    replace star(* 0.1 ** 0.05 *** 0.01) ///
    b(3) se(3) ///
    stats(N r2 first_F, fmt(0 3 2) label("N" "R2" "First-stage F")) ///
    title("IV 估计：制度质量对人均收入的影响（AJR 2001 设计）") ///
    mtitles("OLS" "First Stage" "Reduced Form" "2SLS")

* ---------------------------------------------------------------
* 8. 稳健性检验
* ---------------------------------------------------------------
* 8.1 LIML（有限信息极大似然）
ivreg2 logpgdp lat_abst africa (instit = logmort), liml robust
estimates store liml

* 8.2 剔除非洲国家
ivreg2 logpgdp lat_abst (instit = logmort) if !africa, robust
estimates store iv_noafrica

esttab iv_2sls liml iv_noafrica using "$out/table_iv_robust.csv", ///
    replace star(* 0.1 ** 0.05 *** 0.01) b(3) se(3) ///
    keep(instit) ///
    title("稳健性：IV 估计") ///
    mtitles("基准 2SLS" "LIML" "排除非洲")

di "02_iv_2sls.do 运行完毕"
