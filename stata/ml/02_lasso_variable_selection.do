/* ============================================================
   Stata 对应代码：LASSO 变量筛选
   来源：Belloni, Chernozhukov & Hansen (2014)
   适用 Stata 版本：16+ (内置 lasso 命令)
   ============================================================ */

clear all
set more off

* ── 生成模拟数据 ────────────────────────────────────────────
set seed 42
set obs 500

* 生成 20 个候选变量（真实只有 x1-x5 有效）
forvalues i = 1/20 {
    gen x`i' = rnormal(0, 1)
}

* DGP：只有 x1-x5 影响 y
gen y = 1.2*x1 - 0.8*x2 + 0.5*x3 + 1.0*x4 - 0.6*x5 + rnormal(0, 1.5)

* ── LASSO 变量筛选（Stata 16+ 内置）────────────────────────
lasso linear y x1-x20, nfolds(5)

* 查看选中变量
lassocoef, display(coef, postselection)

* ── Post-LASSO OLS（用选中变量重新估计）────────────────────
* 提取选中变量列表
local selected : colnames e(b)
di "选中变量: `selected'"

* Post-LASSO OLS
reg y `selected', robust
estimates store post_lasso

esttab post_lasso, ///
    b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
    title("Post-LASSO OLS 估计结果") ///
    note("稳健标准误。LASSO 交叉验证选择最优 lambda。")


/* ── Double Selection（Belloni et al. 2014 方法）──────────────
   当研究变量为 d（可能内生），控制变量 x 高维时：
   1. LASSO(y ~ x)，选出 S1
   2. LASSO(d ~ x)，选出 S2
   3. OLS(y ~ d + S1 ∪ S2)
   ──────────────────────────────────────────────────────────── */

* 生成处理变量
gen d = 0.5*x1 + 0.3*x2 + rnormal(0, 1)
replace y = y + 0.4*d   // 真实处理效应 = 0.4

* Double Selection
dsregress y d x1-x20, nfolds(5)
di "Double Selection 处理效应估计："
di "  系数 = " e(b_d) ", SE = " e(se_d)
