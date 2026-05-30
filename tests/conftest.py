"""
pytest 配置文件
- 固定随机种子，保证测试可复现
- 跳过需要 PyTorch / GPU 的测试（如环境无 PyTorch）
"""
import numpy as np
import pytest


def pytest_configure(config):
    np.random.seed(42)


def pytest_collection_modifyitems(items):
    """为需要 torch 的测试自动添加 skipif marker。"""
    try:
        import torch  # noqa: F401
        torch_missing = False
    except ImportError:
        torch_missing = True

    try:
        import gymnasium  # noqa: F401
        gym_missing = False
    except ImportError:
        gym_missing = True

    for item in items:
        # DL 测试需要 torch
        if "dl" in str(item.fspath) or "lstm" in item.name or "tcn" in item.name or "mlp" in item.name:
            if torch_missing:
                item.add_marker(pytest.mark.skip(reason="PyTorch 未安装"))
        # RL 测试需要 torch + gymnasium
        if "rl" in str(item.fspath) or "dqn" in item.name or "portfolio" in item.name:
            if torch_missing or gym_missing:
                item.add_marker(pytest.mark.skip(reason="PyTorch 或 gymnasium 未安装"))
