"""
config - 训练模型相关配置类

Author: 骆昊
Version: 0.0.1
"""
from pathlib import Path
from dataclasses import dataclass

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    """配置类"""
    stopwords_file: Path = BASE_DIR / 'data' / 'stopwords.txt'
    class_file:     Path = BASE_DIR / 'data' / 'tmf_class.txt'

    train_raw_file: Path = BASE_DIR / 'data/raw' / 'tmf_train.txt'
    test_raw_file:  Path = BASE_DIR / 'data/raw' / 'tmf_test.txt'
    valid_raw_file: Path = BASE_DIR / 'data/raw/' / 'tmf_valid.txt'

    train_pre_file: Path = BASE_DIR / 'data/pre' / 'tmf_train.txt'
    test_pre_file:  Path = BASE_DIR / 'data/pre' / 'tmf_test.txt'
    valid_pre_file: Path = BASE_DIR / 'data/pre' / 'tmf_valid.txt'

    model_file:     Path = BASE_DIR / 'models' / 'text-clf-model.pkl'
