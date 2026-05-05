"""
示例3: GARCH模型 - 金融波动率预测

使用GARCH模型对股票收益率的波动率进行建模和预测，
这对风险管理和衍生品定价至关重要。

运行方式: python examples/03_garch_volatility.py
"""

import numpy as np
import pandas as pd
from econml.econometric_models import GARCHModel
import matplotlib.pyplot as plt


def generate_stock_returns():
    """生成模拟的股票日收益率数据（带时变波动率）"""
    np.random.seed(42)
    n_days = 252 * 2  # 2年的交易日
    
    returns = np.zeros(n_days)
    sigma = np.zeros(n_days)
    
    # 初始波动率
    sigma[0] = 0.01
    
    # 参数 (GARCH(1,1))
    omega = 0.00001  # 长期方差
    alpha = 0.08     # ARCH系数
    beta = 0.90      # GARCH系数
    
    for t in range(1, n_days):
        # 条件波动率演化
        sigma[t] = np.sqrt(omega + alpha * returns[t-1]**2 + beta * sigma[t-1]**2)
        
        # 从条件分布生成收益率
        returns[t] = sigma[t] * np.random.standard_normal()
    
    # 创建DataFrame
    dates = pd.date_range(start='2022-01-01', periods=n_days, freq='B')  # B = 工作日
    data = pd.DataFrame({
        'date': dates,
        'returns': returns,
        'true_sigma': sigma
    })
    
    return data


def main():
    print("=" * 60)
    print("示例3: GARCH模型 - 金融波动率预测")
    print("=" * 60)
    
    # 1. 生成数据
    print("\n[第1步] 生成股票收益率数据...")
    data = generate_stock_returns()
    print(f"数据形状: {data.shape}")
    print(f"日期范围: {data['date'].min()} 到 {data['date'].max()}")
    print(f"\n收益率统计:")
    print(f"  均值: {data['returns'].mean():.6f}")
    print(f"  标准差: {data['returns'].std():.6f}")
    print(f"  峰度: {data['returns'].kurtosis():.4f}")
    print(f"  偏度: {data['returns'].skew():.4f}")
    
    # 2. 分割数据
    print("\n[第2步] 分割训练/测试集...")
    train_size = int(0.8 * len(data))
    returns_train = data['returns'].iloc[:train_size].values
    returns_test = data['returns'].iloc[train_size:].values
    dates_test = data['date'].iloc[train_size:]
    
    print(f"  训练集: {len(returns_train)} 个交易日")
    print(f"  测试集: {len(returns_test)} 个交易日")
    
    # 3. 拟合GARCH模型
    print("\n[第3步] 拟合GARCH(1,1)模型...")
    garch = GARCHModel(p=1, q=1)
    garch.fit(returns_train)
    print("GARCH模型拟合完成！")
    
    print(f"\n估计的参数:")
    print(f"  ω (omega): {garch.params[0]:.8f}")
    print(f"  α (alpha): {garch.params[1]:.6f}")
    print(f"  β (beta): {garch.params[2]:.6f}")
    
    # 模型诊断
    alpha_plus_beta = garch.params[1] + garch.params[2]
    print(f"  α + β: {alpha_plus_beta:.6f} {'(稳定)' if alpha_plus_beta < 1 else '(不稳定!)'}")
    
    # 4. 预测波动率
    print("\n[第4步] 预测未来20个交易日的波动率...")
    future_volatility = garch.forecast(returns_train, steps=20)
    
    print(f"\n未来波动率预测:")
    for i in range(min(5, len(future_volatility))):
        print(f"  第{i+1}天: {future_volatility[i]:.6f}")
    
    print(f"  ...(更多预测)")
    for i in range(max(0, len(future_volatility)-5), len(future_volatility)):
        print(f"  第{i+1}天: {future_volatility[i]:.6f}")
    
    # 5. 回测（测试集上的预测）
    print("\n[第5步] 在测试集上进行回测...")
    
    # 计算测试集上的条件波动率
    garch_test = GARCHModel(p=1, q=1)
    garch_test.fit(returns_train)  # 用训练数据拟合
    
    # 逐步预测 (rolling forecast)
    predicted_vol = np.zeros(len(returns_test))
    for i in range(len(returns_test)):
        # 用前面的数据预测
        test_data = np.concatenate([returns_train, returns_test[:i]])
        garch_test.fit(test_data)
        predicted_vol[i] = garch_test.forecast(test_data, steps=1)[0]
    
    # 实现波动率（realized volatility）
    window = 20  # 20日滚动窗口
    realized_vol = np.array([returns_test[max(0, i-window):i].std() 
                             for i in range(1, len(returns_test)+1)])
    
    # 性能指标
    mse_pred = np.mean((realized_vol - predicted_vol) ** 2)
    mae_pred = np.mean(np.abs(realized_vol - predicted_vol))
    
    print(f"  预测波动率 vs 实现波动率:")
    print(f"    MSE: {mse_pred:.8f}")
    print(f"    MAE: {mae_pred:.6f}")
    
    # 6. 应用：VaR计算
    print("\n[第6步] 应用: Value-at-Risk (VaR) 计算...")
    
    current_vol = garch.conditional_volatility[-1]
    confidence_levels = [0.90, 0.95, 0.99]
    z_scores = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}  # 标准正态分布
    
    print(f"\n假设投资组合价值: ¥100,000")
    print(f"当前估计波动率: {current_vol:.4f}")
    print(f"\n一天持有期的VaR估计:")
    
    for conf in confidence_levels:
        var = 100000 * current_vol * z_scores[conf]
        print(f"  {conf*100:.0f}% 置信度: ¥{var:,.2f}")
    
    # 7. 可视化
    print("\n[第7步] 生成可视化...")
    fig = plt.figure(figsize=(14, 10))
    
    # 收益率时间序列
    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(data['date'], data['returns']*100, linewidth=0.8)
    ax1.set_ylabel('收益率 (%)')
    ax1.set_title('日收益率时间序列')
    ax1.grid(True, alpha=0.3)
    
    # 条件波动率（训练集）
    ax2 = plt.subplot(4, 1, 2)
    ax2.plot(data['date'][:train_size], garch.conditional_volatility*100, label='GARCH条件波动率', linewidth=1.5)
    ax2.set_ylabel('波动率 (%)')
    ax2.set_title('训练集：GARCH模型估计的波动率')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 测试集：预测 vs 实现
    ax3 = plt.subplot(4, 1, 3)
    ax3.plot(dates_test, realized_vol*100, label='实现波动率', linewidth=2, alpha=0.7)
    ax3.plot(dates_test, predicted_vol*100, label='GARCH预测波动率', linewidth=2, alpha=0.7)
    ax3.set_ylabel('波动率 (%)')
    ax3.set_title('测试集：预测波动率 vs 实现波动率')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 向前预测
    ax4 = plt.subplot(4, 1, 4)
    last_50_dates = data['date'].iloc[-50:]
    forecast_dates = pd.date_range(start=data['date'].iloc[-1], periods=len(future_volatility)+1, freq='B')[1:]
    
    ax4.plot(last_50_dates, garch.conditional_volatility[-50:]*100, label='历史波动率', linewidth=2)
    ax4.plot(forecast_dates, future_volatility*100, 'r--', label='20天预测', linewidth=2)
    ax4.set_ylabel('波动率 (%)')
    ax4.set_xlabel('时间')
    ax4.set_title('向前20个交易日的波动率预测')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('example_garch_volatility.png', dpi=100)
    print("图表已保存为 example_garch_volatility.png")
    plt.show()
    
    print("\n" + "=" * 60)
    print("示例3执行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
