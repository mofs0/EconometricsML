/*==============================================================
  文件名：08_event_study.do
  模型：  事件研究法（Event Study，市场模型）
  描述：  股权质押公告的股价市场反应（超额收益 AR / CAR）
  依赖：  estout / esttab（ssc install estout）
  数据：  合成日度股票收益率（模拟陈超 & 何佳 2019 设计）
  参考：  陈超, 何佳 (2019). 控股股东股权质押与上市公司信息披露.
          《管理世界》35(3), 168–183.
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
* 1. 生成合成数据（N=200 事件，事件窗口 [-10, 10]）
* ---------------------------------------------------------------
local N_events 200
local T_window 21       // -10 to +10
local est_start -120
local est_end   -21

set obs `= `N_events' * `T_window''
gen event_id  = ceil(_n / `T_window')
gen rel_day   = mod(_n - 1, `T_window') - 10   // -10..+10

* 模拟市场模型参数（alpha=0, beta~1）
by event_id, sort: gen alpha_i = rnormal(0, 0.001) if _n==1
by event_id:       replace alpha_i = alpha_i[1]
by event_id, sort: gen beta_i  = rnormal(1, 0.2)  if _n==1
by event_id:       replace beta_i  = beta_i[1]

* 市场收益率
gen r_mkt = rnormal(0.0005, 0.012)

* 个股收益率：在 rel_day==0 时加入真实效应 -0.5%
gen true_ar  = (rel_day == 0) * (-0.005)
gen r_firm   = alpha_i + beta_i * r_mkt + true_ar + rnormal(0, 0.018)

* 预期收益率（需要估计期参数，此处简化：用 OLS 斜率均值代替）
gen r_exp  = alpha_i + beta_i * r_mkt
gen ar     = r_firm - r_exp       // 超额收益

label var ar      "超额收益率（AR）"
label var rel_day "事件窗口相对日"
label var r_mkt   "市场收益率"
label var r_firm  "个股收益率"

* ---------------------------------------------------------------
* 2. 逐期 AR 均值与 t 检验
* ---------------------------------------------------------------
* 截面均值
collapse (mean) ar_mean=ar (sd) ar_sd=ar (count) ar_n=ar, by(rel_day)
gen ar_se = ar_sd / sqrt(ar_n)
gen t_stat = ar_mean / ar_se
gen p_value = 2 * ttail(ar_n - 1, abs(t_stat))

list rel_day ar_mean ar_se t_stat p_value, noobs sep(0)

* ---------------------------------------------------------------
* 3. 累积超额收益（CAR）
* ---------------------------------------------------------------
sort rel_day
gen car = sum(ar_mean)

* 典型事件窗口
sum car if rel_day == 0    // CAR(-10, 0)
sum car if rel_day == 1    // CAR(-10, +1)
sum car if rel_day == 10   // CAR(-10, +10)

* ---------------------------------------------------------------
* 4. 可视化
* ---------------------------------------------------------------
twoway (bar ar_mean rel_day, barwidth(0.6) color(gs10)) ///
       (rcap ar_mean + 1.96*ar_se ar_mean - 1.96*ar_se rel_day, lc(red)), ///
       xline(0, lp(dash)) yline(0) ///
       legend(label(1 "AR") label(2 "95% CI")) ///
       title("逐日超额收益率（事件窗口 [-10,+10]）") ///
       xtitle("事件日相对天数") ytitle("超额收益率")

twoway line car rel_day, ///
    xline(0, lp(dash)) yline(0) ///
    title("累积超额收益率（CAR）") ///
    xtitle("事件日相对天数") ytitle("CAR")

* ---------------------------------------------------------------
* 5. 回归形式的事件研究（验证）
* ---------------------------------------------------------------
* 重新加载个体数据
clear
set obs `= `N_events' * `T_window''
gen event_id  = ceil(_n / `T_window')
gen rel_day   = mod(_n - 1, `T_window') - 10
by event_id, sort: gen alpha_i = rnormal(0, 0.001) if _n==1
by event_id: replace alpha_i = alpha_i[1]
by event_id, sort: gen beta_i = rnormal(1, 0.2) if _n==1
by event_id: replace beta_i = beta_i[1]
gen r_mkt  = rnormal(0.0005, 0.012)
gen true_ar= (rel_day == 0) * (-0.005)
gen r_firm = alpha_i + beta_i * r_mkt + true_ar + rnormal(0, 0.018)
gen ar     = r_firm - alpha_i - beta_i * r_mkt

* 以 -10 期为基准，生成逐日虚拟变量的交互
forvalues d = -9(1)10 {
    local dname = `d' + 10
    gen day_`dname' = (rel_day == `d')
}
reg ar day_1-day_20, robust
* day_10 对应 rel_day=0，系数应显著为负

di "08_event_study.do 运行完毕"
