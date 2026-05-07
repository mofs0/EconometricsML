/*==============================================================
  文件名：01_ols.do
  模型：  Ordinary Least Squares (OLS) — Mincer 工资方程
  描述：  复现 Mincer (1974) 工资方程，包含基准估计、
          稳健标准误、异方差检验与多规格稳健性分析。
  对应：  empirlab/traditional/ols.py  |  notebooks/T01_OLS_Mincer工资方程.ipynb
  依赖：  Stata 17+；外部命令：estout（ssc install estout）
          可选：outreg2（ssc install outreg2）
  数据：  合成 Mincer 数据（本脚本内部生成，无需外部文件）
  作者：  （使用者填写）
  更新：  2025-05
==============================================================*/


* ---------------------------------------------------------------
* 0. 全局设置
* ---------------------------------------------------------------
clear all
set more off
set seed 42

* ── 路径（使用者修改）──
* global root "C:/Users/yourname/EconometricsML"
* global out  "$root/output"

* ── 若不设置路径，结果输出到当前目录 ──
capture mkdir output
global out "./output"


* ---------------------------------------------------------------
* 1. 数据生成（合成 Mincer 数据，n=500）
*    DGP: ln_wage = 1.2 + 0.08*educ + 0.04*exper - 0.0007*exper^2 + eps
*         eps ~ N(0, 0.35^2)
* ---------------------------------------------------------------
set obs 500

* 受教育年限（6~19 年，离散均匀）
gen educ  = floor(runiform() * 14) + 6

* 工作经验（0~39 年，离散均匀）
gen exper = floor(runiform() * 40)

* 经验平方项
gen exper2 = exper^2

* 扰动项
gen eps = rnormal(0, 0.35)

* 因变量：工资对数
gen ln_wage = 1.2 + 0.08 * educ + 0.04 * exper - 0.0007 * exper2 + eps

* 原始工资水平
gen wage = exp(ln_wage)

label var educ    "Education (years)"
label var exper   "Potential Experience (years)"
label var exper2  "Experience Squared"
label var ln_wage "Log Wage"
label var wage    "Wage"

* 查看描述统计
summarize educ exper exper2 ln_wage wage


* ---------------------------------------------------------------
* 2. 数据预处理
* ---------------------------------------------------------------

* 2.1 检查缺失值
misstable summarize

* 2.2 相关矩阵
correlate ln_wage educ exper exper2

* 2.3 单变量分布（可选，取消注释查看）
* histogram ln_wage, normal title("Distribution of ln(wage)")
* histogram educ,    discrete title("Distribution of Education")


* ---------------------------------------------------------------
* 3. 核心估计
* ---------------------------------------------------------------

* ── 规格 1：仅控制教育年限 ──
eststo m1: regress ln_wage educ
di "R-squared: " e(r2)

* ── 规格 2：教育 + 经验（不含二次项）──
eststo m2: regress ln_wage educ exper

* ── 规格 3：完整 Mincer 方程（基准，经典 OLS 标准误）──
eststo m3: regress ln_wage educ exper exper2

* 查看系数
di "educ  系数: " _b[educ]
di "exper 系数: " _b[exper]
di "exper2系数: " _b[exper2]

* 经验峰值（令 d(ln_wage)/d(exper) = b_exper + 2*b_exper2*exper = 0）
di "经验峰值 = " (-_b[exper] / (2 * _b[exper2])) " 年"


* ── 规格 4：完整 Mincer 方程（HC1 稳健标准误）──
eststo m4: regress ln_wage educ exper exper2, robust


* ---------------------------------------------------------------
* 4. 推断结果表（输出到屏幕）
* ---------------------------------------------------------------

esttab m1 m2 m3 m4,                    ///
    title("OLS Estimates: Mincer Wage Equation")     ///
    mtitle("Only educ" "educ+exper" "Mincer OLS" "Mincer HC1")  ///
    b(4) se(4) star(* 0.10 ** 0.05 *** 0.01)        ///
    stats(N r2 r2_a, fmt(%9.0f %9.4f %9.4f)         ///
          labels("N" "R²" "adj-R²"))                 ///
    keep(educ exper exper2 _cons)                    ///
    order(educ exper exper2 _cons)                   ///
    note("Standard errors in parentheses." ///
         "Columns 1-3: OLS SE; Column 4: HC1 robust SE.")


* ── 导出 CSV（可选）──
* esttab m1 m2 m3 m4 using "$out/01_ols_results.csv", replace ///
*     b(4) se(4) star(* 0.10 ** 0.05 *** 0.01)


* ---------------------------------------------------------------
* 5. 稳健性检验
* ---------------------------------------------------------------

* 5.1 残差诊断：预测值与残差
quietly regress ln_wage educ exper exper2
predict y_hat,   xb
predict resid,   residuals

* 残差基本统计
summarize resid
di "残差均值（应≈0）: " r(mean)

* 5.2 Breusch-Pagan 异方差检验
* H0: 残差方差为常数（同方差）
quietly regress ln_wage educ exper exper2
estat hettest
di "若 p < 0.05，拒绝同方差假设，应使用稳健标准误（HC1）"

* 5.3 White 检验
estat imtest, white
di "White 检验：若 p < 0.05，存在异方差"

* 5.4 VIF 多重共线性检验
quietly regress ln_wage educ exper exper2
estat vif
di "VIF > 10 通常表示严重多重共线性（exper 与 exper2 相关性高属正常）"

* 5.5 RESET 检验（函数形式误设检验）
ovtest
di "RESET：若 p < 0.05，提示模型函数形式可能存在误设"

* 5.6 不同样本量稳健性
* （生产环境中通常对不同子样本或年份循环，此处展示不同 n 的思路）
foreach n_sub in 100 200 300 {
    quietly {
        preserve
        keep in 1/`n_sub'
        regress ln_wage educ exper exper2
        local b_educ = string(_b[educ], "%7.4f")
        local r2     = string(e(r2), "%7.4f")
        restore
    }
    di "n=`n_sub' | educ coef=`b_educ' | R2=`r2'"
}


* ---------------------------------------------------------------
* 6. 可视化（取消注释运行）
* ---------------------------------------------------------------

* 6.1 残差分布直方图
* histogram resid, normal title("Residual Distribution") xtitle("Residual")

* 6.2 残差 vs 拟合值散点图（异方差目视检查）
* twoway scatter resid y_hat, ///
*     yline(0) title("Residuals vs Fitted") ///
*     ytitle("Residual") xtitle("Fitted ln(wage)")

* 6.3 Mincer 曲线：固定 educ=12，经验轨迹
* quietly regress ln_wage educ exper exper2
* local b0     = _b[_cons]
* local b_educ = _b[educ]
* local b_exp  = _b[exper]
* local b_exp2 = _b[exper2]
* twoway function y = `b0' + `b_educ'*12 + `b_exp'*x + `b_exp2'*x^2, ///
*     range(0 40) title("Mincer Curve (educ=12 fixed)") ///
*     xtitle("Experience (years)") ytitle("Predicted ln(wage)")


* ---------------------------------------------------------------
* 收尾
* ---------------------------------------------------------------
di "01_ols.do 执行完毕。"
di "核心结果：完整 Mincer 方程见规格 m3（OLS）和 m4（HC1 稳健）。"
