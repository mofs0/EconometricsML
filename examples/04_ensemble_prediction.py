"""
示例4: 集成模型 - 股票收益率预测

使用Random Forest和Gradient Boosting的加权集成方法，
预测股票收益率，展示现代机器学习在金融中的应用。

运行方式: python examples/04_ensemble_prediction.py
"""

import numpy as np
import pandas as pd
from econml.ml_models import EnsemblePredictor
from econml.utils import calculate_metrics, directional_accuracy
import matplotlib.pyplot as plt


def create_financial_features(prices, lookback=10, indicators=True):
    """
    创建用于预测的金融特征
    
    参数:
        prices: 价格序列
        lookback: 回看窗口
        indicators: 是否添加技术指标
    """
    returns = np.diff(np.log(prices))
    
    # 基础特征：滞后收益率
    X = np.zeros((len(returns) - lookback, lookback))
    for i in range(len(X)):
        X[i] = returns[i:i+lookback]
    
    if indicators:
        # 添加技术指标特征
        sma = np.zeros(len(prices) - 1)
        for i in range(lookback, len(returns)):
            sma[i] = np.mean(prices[i-lookback:i])
        
        # 价格动量
        momentum = np.diff(prices[lookback-1:]) / prices[lookback-1:-1]
        
        # 添加这些特征
        X = np.column_stack([X, sma[lookback:len(X)+lookback], momentum[:len(X)]])
    
    # 目标：下期收益率
    y = returns[lookback+1:]
    
    return X, y


def generate_financial_data(n_days=500):
    """生成带有趋势和波动的模拟股票价格"""
    np.random.seed(42)
    
    returns = np.random.normal(0.0005, 0.02, n_days)
    # 添加一些自相关性（momentum effect）
    returns = 0.6 * np.roll(returns, 1) + 0.4 * returns
    
    prices = np.exp(np.cumsum(returns))
    prices = prices * 100  # 缩放到合理的价格水平
    
    dates = pd.date_range(start='2022-01-01', periods=n_days, freq='D')
    
    data = pd.DataFrame({
        'date': dates,
        'price': prices,
        'returns': np.concatenate([[np.nan], returns])
    })
    
    return data


def main():
    print("=" * 70)
    print("示例4: 集成模型 - 金融时间序列预测")
    print("=" * 70)
    
    # 1. 生成数据
    print("\n[第1步] 生成金融时间序列数据...")
    data = generate_financial_data(n_days=500)
    print(f"数据形状: {data.shape}")
    print(f"日期范围: {data['date'].min()} 到 {data['date'].max()}")
    print(f"\n价格统计:")
    print(f"  最小: ¥{data['price'].min():.2f}")
    print(f"  最大: ¥{data['price'].max():.2f}")
    print(f"  均值: ¥{data['price'].mean():.2f}")
    
    # 2. 特征工程
    print("\n[第2步] 创建特征...")
    X, y = create_financial_features(data['price'].values, lookback=10, indicators=True)
    print(f"特征形状: {X.shape}")
    print(f"目标形状: {y.shape}")
    
    # 3. 分割数据
    print("\n[第3步] 分割训练/测试集...")
    train_ratio = 0.7
    train_size = int(train_ratio * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    
    print(f"  训练集: {X_train.shape[0]} 个样本")
    print(f"  测试集: {X_test.shape[0]} 个样本")
    
    # 4. 训练集成模型
    print("\n[第4步] 训练集成模型...")
    ensemble = EnsemblePredictor(n_estimators=100, method='voting')
    ensemble.fit(X_train, y_train)
    print("集成模型训练完成！")
    
    # 5. 预测
    print("\n[第5步] 进行预测...")
    y_pred_train = ensemble.predict(X_train)
    y_pred_test = ensemble.predict(X_test)
    
    # 6. 评估
    print("\n[第6步] 模型评估...")
    
    train_metrics = calculate_metrics(y_train, y_pred_train)
    test_metrics = calculate_metrics(y_test, y_pred_test)
    
    print(f"\n训练集指标:")
    for metric_name, value in train_metrics.items():
        if metric_name == 'MAPE':
            print(f"  {metric_name}: {value:.2f}%")
        else:
            print(f"  {metric_name}: {value:.6f}")
    
    print(f"\n测试集指标:")
    for metric_name, value in test_metrics.items():
        if metric_name == 'MAPE':
            print(f"  {metric_name}: {value:.2f}%")
        else:
            print(f"  {metric_name}: {value:.6f}")
    
    # 方向准确度（对交易者重要）
    train_dir_acc = directional_accuracy(y_train, y_pred_train)
    test_dir_acc = directional_accuracy(y_test, y_pred_test)
    
    print(f"\n方向准确度（上升/下降预测正确率）:")
    print(f"  训练集: {train_dir_acc:.2%}")
    print(f"  测试集: {test_dir_acc:.2%}")
    
    # 7. 特征重要性
    print("\n[第7步] 特征重要性分析...")
    feature_importance = ensemble.feature_importance()
    
    feature_names = [f'lag_{i+1}' for i in range(10)]
    if X.shape[1] > 10:
        feature_names.extend(['SMA', 'Momentum'])
    
    top_k = 5
    top_indices = np.argsort(feature_importance)[-top_k:][::-1]
    
    print(f"\n前{top_k}个重要特征:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank}. {feature_names[idx]}: {feature_importance[idx]:.4f}")
    
    # 8. 交易信号生成
    print("\n[第8步] 生成交易信号...")
    
    # 基于预测方向的简单交易策略
    buy_signal = y_pred_test > 0
    actual_positive = y_test > 0
    
    strategy_returns = np.where(buy_signal == actual_positive, 
                                 np.abs(y_test), 
                                 -np.abs(y_test))
    
    cumulative_returns = np.cumprod(1 + strategy_returns)
    total_return = cumulative_returns[-1] - 1
    sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
    
    print(f"\n策略表现 (基于预测信号):")
    print(f"  累计收益率: {total_return:.2%}")
    print(f"  夏普比率: {sharpe_ratio:.4f}")
    print(f"  胜率: {np.mean(strategy_returns > 0):.2%}")
    
    # 9. 可视化
    print("\n[第9步] 生成可视化...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # 实际值 vs 预测值
    test_dates = data['date'].iloc[train_size+10:]  # 对应X_test的日期
    
    axes[0, 0].scatter(y_test, y_pred_test, alpha=0.5, s=20)
    axes[0, 0].plot([-0.05, 0.05], [-0.05, 0.05], 'r--', lw=2)
    axes[0, 0].set_xlabel('实际收益率')
    axes[0, 0].set_ylabel('预测收益率')
    axes[0, 0].set_title('实际 vs 预测 收益率')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 残差
    residuals = y_test - y_pred_test
    axes[0, 1].scatter(y_pred_test, residuals, alpha=0.5, s=20)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('预测值')
    axes[0, 1].set_ylabel('残差')
    axes[0, 1].set_title('残差图')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 特征重要性
    axes[1, 0].barh(feature_names, feature_importance)
    axes[1, 0].set_xlabel('重要性得分')
    axes[1, 0].set_title('特征重要性')
    
    # 累计收益
    axes[1, 1].plot(test_dates, cumulative_returns, linewidth=2, label='策略收益')
    axes[1, 1].axhline(y=1, color='r', linestyle='--', label='基准')
    axes[1, 1].set_xlabel('时间')
    axes[1, 1].set_ylabel('累计收益倍数')
    axes[1, 1].set_title('策略累计收益曲线')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('example_ensemble_prediction.png', dpi=100)
    print("图表已保存为 example_ensemble_prediction.png")
    plt.show()
    
    print("\n" + "=" * 70)
    print("示例4执行完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
