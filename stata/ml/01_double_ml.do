/*==============================================================
  文件名：01_double_ml.do
  模型：  Double Machine Learning (近似双重交叉拟合流程)
  描述：  使用 Lasso 控制高维变量，估计处理变量 D 对结果 Y 的因果效应
          近似 Chernozhukov et al. (2018) 的 DML 估计量
  依赖：  Stata 17+ 内置 lasso 命令 (elasticnet / lassoregress)
          esttab (ssc install estout, 可选)
  数据：  合成截面数据（N=500，可替换为真实 dta 文件）
  参考：  Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E.,
          Hansen, C., Newey, W., & Robins, J. (2018).
          Double/debiased machine learning. Econometrics Journal, 21(1).
  作者：  （使用者填写）
  更新：  2025-05
==============================================================*/

* ---------------------------------------------------------------
* 0. 全局设置（使用者修改路径）
* ---------------------------------------------------------------
global root  "C:/Users/yourname/EconometricsML"
global out   "$root/output"

clear all
set more off
set seed 42
capture mkdir "$out"

* ---------------------------------------------------------------
* 1. 生成合成数据（N=500）
* ---------------------------------------------------------------
* DGP: Y = 1.5*D + X*beta + eps，D 受 X 影响（存在混淆）
set obs 500
forvalues j = 1/10 {
    gen x`j' = rnormal()
}
gen d = 0.8*x1 + 0.5*x2 - 0.4*x3 + 0.3*x4 + 0.2*x5 + rnormal()
gen y = 1.5*d + 0.6*x1 - 0.5*x2 + 0.3*x3 + 0.4*x6 + rnormal()

label var y "结果变量"
label var d "处理变量（真实效应=1.5）"

* ---------------------------------------------------------------
* 2. 朴素 OLS（基准，存在混淆偏差）
* ---------------------------------------------------------------
reg y d x1-x10, robust
estimates store ols_naive

* ---------------------------------------------------------------
* 3. DML 近似（单次残差化）
*    A: Lasso 拟合 Y ~ X，得残差 Y_tilde
*    B: Lasso 拟合 D ~ X，得残差 D_tilde
*    C: OLS Y_tilde ~ D_tilde（部分线性回归）
* ---------------------------------------------------------------
lassoregress y x1-x10, selection(cv) rseed(42)
predict double yhat, xb
gen double y_tilde = y - yhat

lassoregress d x1-x10, selection(cv) rseed(42)
predict double dhat, xb
gen double d_tilde = d - dhat

reg y_tilde d_tilde, robust noconstant
estimates store dml_approx

* ---------------------------------------------------------------
* 4. DML 2-折交叉拟合（更标准的实现）
* ---------------------------------------------------------------
gen fold = mod(_n, 2) + 1
gen double y_tilde_cv = .
gen double d_tilde_cv = .

forvalues k = 1/2 {
    local other = 3 - `k'

    quietly lassoregress y x1-x10 if fold == `other', selection(cv) rseed(42)
    quietly predict double tmp_yhat, xb
    quietly replace y_tilde_cv = y - tmp_yhat if fold == `k'
    drop tmp_yhat

    quietly lassoregress d x1-x10 if fold == `other', selection(cv) rseed(42)
    quietly predict double tmp_dhat, xb
    quietly replace d_tilde_cv = d - tmp_dhat if fold == `k'
    drop tmp_dhat
}

reg y_tilde_cv d_tilde_cv, robust noconstant
estimates store dml_cv2

* ---------------------------------------------------------------
* 5. 结果汇总与输出
* ---------------------------------------------------------------
display ""
display "======= 方法对比（真实效应=1.5）======="
estimates table ols_naive dml_approx dml_cv2, ///
    keep(d d_tilde d_tilde_cv) b(%8.4f) se(%8.4f) ///
    title("OLS vs DML 近似 vs DML 2-折交叉拟合")

capture which esttab
if _rc == 0 {
    esttab ols_naive dml_approx dml_cv2 using "$out/01_double_ml.csv", ///
        replace b(4) se(4) title("Double ML 结果对比") ///
        mtitles("OLS" "DML_近似" "DML_CV2")
}

* ---------------------------------------------------------------
* 6. 稳健性：ElasticNet 替代 Lasso
* ---------------------------------------------------------------
elasticnet linear y x1-x10, selection(cv) rseed(42) alpha(0.5)
predict double yhat_en, xb
gen double y_tilde_en = y - yhat_en

elasticnet linear d x1-x10, selection(cv) rseed(42) alpha(0.5)
predict double dhat_en, xb
gen double d_tilde_en = d - dhat_en

reg y_ti