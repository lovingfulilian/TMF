"""
config - 项目配置类
"""
from pathlib import Path
from dataclasses import dataclass

# 获取项目根目录
BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    """配置类"""
    train_datapath: Path = BASE_DIR / 'data/tmf_train.txt'
    test_datapath:  Path = BASE_DIR / 'data/tmf_test.txt'
    dev_datapath:   Path = BASE_DIR / 'data/tmf_dev.txt'
    class_datapath: Path = BASE_DIR / 'data/tfm_class.txt'
