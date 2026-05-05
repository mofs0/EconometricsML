"""
示例1: OLS回归 - GDP增长预测

这个示例展示如何使用OLS模型进行经济预测，
包括数据加载、模型拟合、统计检验和结果解释。

运行方式: python examples/01_ols_regression.py
"""

import numpy as np
import pandas as pd
from econml.econometric_models import OLSRegression
from econml.utils import calculate_metrics, plot_time_series
import matplotlib.pyplot as plt


def generate_sample_data():
    """生成示例经济数据"""
    np.random.seed(42)
    n_samples = 100
    
    # 生成自变量
    inflation = np.random.uniform(1.5, 4.0, n_samples)
    unemployment = np.random.uniform(3.5, 7.0, n_samples)
    interest_rate = np.random.uniform(1.0, 5.0, n_samples)
    trade_balance = np.random.uniform(-50, 50, n_samples)
    
    # 生成因变量 (真实模型 + 噪声)
    # GDP增长 = 0.5*inflation - 0.3*unemployment + 0.4*interest_rate - 0.01*trade_balance + 噪声
    gdp_growth = (
        0.5 * inflation 
        - 0.3 * unemployment 
        + 0.4 * interest_rate 
        - 0.01 * trade_balance 
        + np.random.normal(0, 0.5, n_samples)
    )
    
    # 创建DataFrame
    data = pd.DataFrame({
        'inflation': inflation,
        'unemployment': unemployment,
        'interest_rate': interest_rate,
        'trade_balance': trade_balance,
        'gdp_growth': gdp_growth,
        'date': pd.date_range(start='2015-01-01', periods=n_samples, freq='M')
    })
    
    return data


def main():
    print("=" * 60)
    print("示例1: OLS回归 - GDP增长预测")
    print("=" * 60)
    
    # 1. 生成数据
    print("\n[第1步] 加载数据...")
    data = generate_sample_data()
    print(f"数据形状: {data.shape}")
    print(f"变量列表: {data.columns.tolist()}")
    
    # 2. 准备特征和目标
    X = data[['inflation', 'unemployment', 'interest_rate', 'trade_balance']].values
    y = data['gdp_growth'].values
    
    # 分割数据（简单分割，实际应使用交叉验证）
    train_size = int(0.8 * len(data))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_train_size], y[train_size:]
    
    # 3. 拟合模型
    print("\n[第2步] 拟合OLS模型...")
    model = OLSRegression(fit_intercept=True)
    model.fit(X_train, y_train)
    print("模型拟合完成！")
    
    # 4. 获取统计结果
    print("\n[第3步] 回归结果统计...")
    summary = model.summary()
    
    print(f"\n基本统计量:")
    print(f"  样本量: {summary['n_observations']}")
    print(f"  特征数: {summary['n_features']}")
    print(f"  R²: {summary['r_squared']:.4f}")
    print(f"  调整R²: {summary['adjusted_r_squared']:.4f}")
    
    print(f"\n回归系数:")
    feature_names = ['inflation', 'unemployment', 'interest_rate', 'trade_balance']
    for i, (name, coef) in enumerate(zip(feature_names, summary['coefficients'])):
        p_val = summary['p_values'][i+1]  # +1因为第一个p值是截距项
        sig = '***' if p_val < 0.01 else ('**' if p_val < 0.05 else ('*' if p_val < 0.1 else ''))
        print(f"  {name}: {coef:7.4f}  (p-value: {p_val:.4f}) {sig}")
    
    print(f"\n截距项: {summary['intercept']:.4f}")
    print(f"  (p-value: {summary['p_values'][0]:.4f})")
    
    # 5. 预测
    print("\n[第4步] 进行预测...")
    y_pred = model.predict(X_test)
    
    # 6. 模型评估
    print("\n[第5步] 模型评估...")
    metrics = calculate_metrics(y_test, y_pred)
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAE: {metrics['MAE']:.4f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    print(f"  R²: {metrics['R2']:.4f}")
    
    # 7. 可视化
    print("\n[第6步] 生成可视化...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 实际值 vs 预测值
    axes[0, 0].scatter(y_test, y_pred, alpha=0.6)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('实际GDP增长')
    axes[0, 0].set_ylabel('预测GDP增长')
    axes[0, 0].set_title('实际值 vs 预测值')
    
    # 残差
    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('预测值')
    axes[0, 1].set_ylabel('残差')
    axes[0, 1].set_title('残差图')
    
    # 残差分布
    axes[1, 0].hist(residuals, bins=20, edgecolor='black')
    axes[1, 0].set_xlabel('残差')
    axes[1, 0].set_ylabel('频率')
    axes[1, 0].set_title('残差分布')
    
    # 特征与目标的关系
    axes[1, 1].scatter(data['inflation'][:train_size], y_train, label='训练数据', alpha=0.6)
    axes[1, 1].scatter(data['inflation'][train_size:], y_test, label='测试数据', alpha=0.6)
    axes[1, 1].set_xlabel('通货膨胀率')
    axes[1, 1].set_ylabel('GDP增长')
    axes[1, 1].set_title('通胀与GDP增长的关系')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('example_ols_results.png', dpi=100)
    print("图表已保存为 example_ols_results.png")
    plt.show()
    
    print("\n" + "=" * 60)
    print("示例1执行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
