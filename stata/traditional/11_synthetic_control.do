/*==============================================================
  文件名：11_synthetic_control.do
  模型：  合成控制法（Synthetic Control Method, SCM）
  描述：  政策冲击对地区经济增长的因果效应
  依赖：  synth（ssc install synth）
          synth_runner（ssc install synth_runner）
  数据：  合成省级面板数据（模拟 Abadie et al. 2010 设计）
  参考：  Abadie, A., Diamond, A., & Hainmueller, J. (2010).
          Synthetic Control Methods for Comparative Case Studies.
          JASA, 105(490), 493–505.
          纪洋, 王鹏飞等 (2018). 资本账户开放的增长效应.
          《经济学(季刊)》18(2).
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
* 1. 生成合成省级面板数据
* ---------------------------------------------------------------
* 11 个省（1 个处置 + 10 个捐赠），年份 2000–2019
* 干预：处置省 2010 年获得政策优惠（特区设立等）

local N_prov 11
local T      20
local T_pre  10      // 2000–2009
local treat_yr 2010

set obs `= `N_prov' * `T''
gen prov_id = ceil(_n / `T')
gen year    = 1999 + mod(_n - 1, `T') + 1   // 2000..2019
gen treated = (prov_id == 1)

* 公共因子
by year, sort: gen beta_t = sum(rnormal(0.3, 0.15)) if _n==1
by year: replace beta_t = beta_t[1]

* 个体特征
by prov_id, sort: gen mu_i    = runiform(0.5, 1.5) if _n==1
by prov_id: replace  mu_i    = mu_i[1]
by prov_id, sort: gen alpha_i = rnormal(5, 1) if _n==1
by prov_id: replace  alpha_i = alpha_i[1]

* 结果变量（GDP 增长率）
gen gdp_growth = alpha_i + beta_t * mu_i + rnormal(0, 0.4)
* 处置效应（线性增长，最大 +3 个百分点）
replace gdp_growth = gdp_growth + (year - `treat_yr' + 1) * 0.3 ///
    if treated & year >= `treat_yr'

* 协变量（前处置期特征）
gen invest_rate = 0.3 + 0.05*mu_i + rnormal(0, 0.02)
gen trade_open  = 0.2 + 0.03*mu_i + rnormal(0, 0.015)

label var gdp_growth  "GDP 增长率（%）"
label var invest_rate "投资率"
label var trade_open  "贸易开放度"
label var treated     "处置省（1=政策省）"

xtset prov_id year

* ---------------------------------------------------------------
* 2. 描述性统计与原始趋势图
* ---------------------------------------------------------------
xtsum gdp_growth if treated
xtsum gdp_growth if !treated

twoway (line gdp_growth year if prov_id == 1, lc(red) lw(medthick)) ///
       (line gdp_growth year if prov_id != 1 & !treated, lc(gs12) lw(thin)), ///
       xline(`treat_yr', lp(dash)) ///
       legend(label(1 "处置省") label(2 "捐赠池")) ///
       title("原始趋势：GDP 增长率") ///
       xtitle("年份") ytitle("GDP 增长率（%）")

* ---------------------------------------------------------------
* 3. 合成控制估计（需安装 synth）
* ---------------------------------------------------------------
* synth 命令语法：
*   synth outcome predictor1 predictor2 ... ,
*       trunit(处置单元ID) trperiod(干预期) xperiod(前处置期) figure

synth gdp_growth                                       ///
    gdp_growth(2000) gdp_growth(2003) gdp_growth(2006) gdp_growth(2009) ///
    invest_rate(2000(1)2009) trade_open(2000(1)2009),  ///
    trunit(1) trperiod(`treat_yr')                     ///
    resultsperiod(2000(1)2019)                         ///
    fig keep("$out/scm_results") replace

* 提取权重
matrix list e(W_weights)
matrix list e(X_balance)

* ---------------------------------------------------------------
* 4. 处置效应序列
* ---------------------------------------------------------------
* synth 结果存储于 e(Y_treated) 和 e(Y_synthetic)
matrix Y_treat = e(Y_treated)
matrix Y_synth = e(Y_synthetic)
matrix gap     = Y_treat - Y_synth

* 转为数据集
svmat gap, name(scm_gap)
rename scm_gap1 gap
gen year2 = 1999 + _n if gap != .
twoway line gap year2, xline(`treat_yr', lp(dash)) yline(0) ///
    title("合成控制：处置效应（Gap）") ///
    xtitle("年份") ytitle("处置效应（百分点）")

* ---------------------------------------------------------------
* 5. 安慰剂检验（In-space placebo）
* ---------------------------------------------------------------
* 逐一将每个捐赠单元视为"处置单元"，比较 gap 分布
* 使用 synth_runner（若已安装）：
* synth_runner gdp_growth                                    ///
*     gdp_growth(2000) gdp_growth(2005) gdp_growth(2009)    ///
*     invest_rate(2000(1)2009) trade_open(2000(1)2009),     ///
*     trunit(1) trperiod(`treat_yr') gen_vars

* 手动安慰剂（以捐赠单元 2 为例）
capture {
    synth gdp_growth                                            ///
        gdp_growth(2000) gdp_growth(2003) gdp_growth(2006) gdp_growth(2009) ///
        invest_rate(2000(1)2009) trade_open(2000(1)2009),      ///
        trunit(2) trperiod(`treat_yr')                         ///
        resultsperiod(2000(1)2019) replace
    matrix gap_placebo = e(Y_treated) - e(Y_synthetic)
    di "安慰剂（prov_id=2）post-period mean gap："
    matrix list gap_placebo
}

* ---------------------------------------------------------------
* 6. RMSPE 比值检验
* ---------------------------------------------------------------
* post_rmspe / pre_rmspe 比值越大，处置效应越显著
* （真实处置省的比值应显著高于大多数捐赠省）

di "11_synthetic_control.do 运行完毕"
