"""
示例2: VAR模型 - 多变量时间序列预测

展示如何使用向量自回归(VAR)模型预测多个经济变量，
比如GDP、通胀率和失业率的联动关系。

运行方式: python examples/02_var_forecasting.py
"""

import numpy as np
import pandas as pd
from econml.econometric_models import VARModel
from econml.utils import plot_time_series
import matplotlib.pyplot as plt


def generate_var_data():
    """生成模拟的多变量时间序列数据"""
    np.random.seed(42)
    n_obs = 150
    
    # 初始值
    gdp = np.zeros(n_obs)
    inflation = np.zeros(n_obs)
    unemployment = np.zeros(n_obs)
    
    gdp[0] = 3.0
    inflation[0] = 2.0
    unemployment[0] = 5.0
    
    # VAR(1)过程: 变量之间存在动态关系
    # 这里模拟真实的宏观经济关系
    for t in range(1, n_obs):
        # GDP受到自身滞后、通胀和失业的影响
        gdp[t] = 0.5 + 0.7 * gdp[t-1] - 0.3 * inflation[t-1] - 0.2 * unemployment[t-1] + np.random.normal(0, 0.2)
        
        # 通胀与GDP和失业有关 (Phillips曲线关系)
        inflation[t] = 1.0 + 0.3 * gdp[t-1] + 0.2 * inflation[t-1] - 0.1 * unemployment[t-1] + np.random.normal(0, 0.15)
        
        # 失业率（NAIRU模型）
        unemployment[t] = 4.5 - 0.15 * gdp[t-1] + 0.8 * unemployment[t-1] + 0.05 * inflation[t-1] + np.random.normal(0, 0.3)
    
    # 创建DataFrame
    data = pd.DataFrame({
        'gdp_growth': gdp,
        'inflation_rate': inflation,
        'unemployment_rate': unemployment,
        'date': pd.date_range(start='2015-01-01', periods=n_obs, freq='Q')
    })
    
    return data


def main():
    print("=" * 60)
    print("示例2: VAR模型 - 多变量时间序列预测")
    print("=" * 60)
    
    # 1. 生成数据
    print("\n[第1步] 生成多变量时间序列数据...")
    data = generate_var_data()
    print(f"数据形状: {data.shape}")
    print(f"时间范围: {data['date'].min()} 到 {data['date'].max()}")
    print("\n前5行数据:")
    print(data.head())
    
    # 2. 可视化原始数据
    print("\n[第2步] 可视化原始时间序列...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    axes[0].plot(data['date'], data['gdp_growth'], marker='o', linestyle='-', markersize=3)
    axes[0].set_ylabel('GDP增长率 (%)')
    axes[0].set_title('GDP增长率时间序列')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(data['date'], data['inflation_rate'], marker='o', linestyle='-', markersize=3, color='orange')
    axes[1].set_ylabel('通胀率 (%)')
    axes[1].set_title('通胀率时间序列')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(data['date'], data['unemployment_rate'], marker='o', linestyle='-', markersize=3, color='red')
    axes[2].set_ylabel('失业率 (%)')
    axes[2].set_xlabel('时间')
    axes[2].set_title('失业率时间序列')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('example_var_timeseries.png', dpi=100)
    print("时间序列图已保存为 example_var_timeseries.png")
    
    # 3. 准备数据
    print("\n[第3步] 准备模型数据...")
    X = data[['gdp_growth', 'inflation_rate', 'unemployment_rate']].values
    
    # 分割为训练集和测试集
    train_size = int(0.8 * len(X))
    X_train = X[:train_size]
    X_test = X[train_size:]
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 4. 拟合VAR模型
    print("\n[第4步] 拟合VAR(1)模型...")
    var_model = VARModel(lag_order=1)
    var_model.fit(X_train)
    print("VAR模型拟合完成！")
    
    # 5. 进行多步预测
    print("\n[第5步] 进行向前12个季度的预测...")
    forecast = var_model.forecast(X_train, steps=12)
    
    print(f"\n预测结果 (前5个季度):")
    print("GDP增长 | 通胀率 | 失业率")
    for i in range(min(5, len(forecast))):
        print(f"{forecast[i, 0]:7.2f} | {forecast[i, 1]:6.2f} | {forecast[i, 2]:7.2f}")
    
    # 6. 评估预测
    print("\n[第6步] 评估预测性能...")
    # 对测试集进行一步步预测
    predictions = []
    current_state = X_train[-1:].copy()
    
    for _ in range(len(X_test)):
        next_pred = var_model.forecast(current_state, steps=1)
        predictions.append(next_pred[0])
        current_state = np.vstack([current_state[1:], next_pred])
    
    predictions = np.array(predictions)
    
    # 计算均方误差
    mse = np.mean((X_test - predictions) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(X_test - predictions))
    
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    
    # 7. 可视化预测结果
    print("\n[第7步] 可视化预测结果...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # 历史数据和测试集日期
    historical_dates = data['date'][:train_size + len(X_test)]
    test_dates = data['date'][train_size:train_size + len(X_test)]
    forecast_dates = pd.date_range(start=test_dates[-1], periods=len(forecast), freq='Q')
    
    # GDP
    axes[0].plot(historical_dates[:train_size], X_train[:, 0], 'b-', label='历史数据', linewidth=2)
    axes[0].plot(test_dates, X_test[:, 0], 'g-', label='测试数据', linewidth=2)
    axes[0].plot(test_dates, predictions[:, 0], 'r--', label='预测值', linewidth=2)
    axes[0].plot(forecast_dates, forecast[:, 0], 'orange', linestyle='--', label='向前预测', linewidth=2)
    axes[0].set_ylabel('GDP增长率 (%)')
    axes[0].set_title('GDP增长率: 实际值 vs 预测值')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 通胀
    axes[1].plot(historical_dates[:train_size], X_train[:, 1], 'b-', label='历史数据', linewidth=2)
    axes[1].plot(test_dates, X_test[:, 1], 'g-', label='测试数据', linewidth=2)
    axes[1].plot(test_dates, predictions[:, 1], 'r--', label='预测值', linewidth=2)
    axes[1].plot(forecast_dates, forecast[:, 1], 'orange', linestyle='--', label='向前预测', linewidth=2)
    axes[1].set_ylabel('通胀率 (%)')
    axes[1].set_title('通胀率: 实际值 vs 预测值')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 失业率
    axes[2].plot(historical_dates[:train_size], X_train[:, 2], 'b-', label='历史数据', linewidth=2)
    axes[2].plot(test_dates, X_test[:, 2], 'g-', label='测试数据', linewidth=2)
    axes[2].plot(test_dates, predictions[:, 2], 'r--', label='预测值', linewidth=2)
    axes[2].plot(forecast_dates, forecast[:, 2], 'orange', linestyle='--', label='向前预测', linewidth=2)
    axes[2].set_ylabel('失业率 (%)')
    axes[2].set_xlabel('时间')
    axes[2].set_title('失业率: 实际值 vs 预测值')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('example_var_forecast.png', dpi=100)
    print("预测结果图已保存为 example_var_forecast.png")
    plt.show()
    
    print("\n" + "=" * 60)
    print("示例2执行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
