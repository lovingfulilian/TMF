"""
config - 训练模型相关配置类

Author: 骆昊
Version: 0.0.1
"""
import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

if not os.path.exists(os.path.join(BASE_DIR, 'models')):
    os.mkdir(os.path.join(BASE_DIR, 'models'))

if not os.path.exists(os.path.join(BASE_DIR, 'models_quantized')):
    os.mkdir(os.path.join(BASE_DIR, 'models_quantized'))

if not os.path.exists(os.path.join(BASE_DIR, 'models_distilled')):
    os.mkdir(os.path.join(BASE_DIR, 'models_distilled'))

if not os.path.exists(os.path.join(BASE_DIR, 'models_pruned')):
    os.mkdir(os.path.join(BASE_DIR, 'models_pruned'))


@dataclass(frozen=True)
class Config:
    """配置类"""
    pretrained_model:    str = 'hfl/chinese-macbert-base'

    stopwords_file:      Path = BASE_DIR / 'data' / 'stopwords.txt'
    class_file:          Path = BASE_DIR / 'data' / 'tmf_class.txt'

    train_raw_file:      Path = BASE_DIR / 'data/raw' / 'tmf_train.txt'
    test_raw_file:       Path = BASE_DIR / 'data/raw' / 'tmf_test.txt'
    valid_raw_file:      Path = BASE_DIR / 'data/raw/' / 'tmf_valid.txt'

    train_pre_file:      Path = BASE_DIR / 'data/pre' / 'tmf_train.txt'
    test_pre_file:       Path = BASE_DIR / 'data/pre' / 'tmf_test.txt'
    valid_pre_file:      Path = BASE_DIR / 'data/pre' / 'tmf_valid.txt'

    pkl_model_file:      Path = BASE_DIR / 'models' / 'text-clf-model.pkl'
    onnx_model_file:     Path = BASE_DIR / 'models' / 'text-clf-model.onnx'
    ftz_model_file:      Path = BASE_DIR / 'models' / 'text-clf-model.ftz'

    model_output_dir:    Path = BASE_DIR / 'models'
    quantized_model_dir: Path = BASE_DIR / 'models_quantized'
    distilled_model_dir: Path = BASE_DIR / 'models_distilled'
    pruned_model_dir:    Path = BASE_DIR / 'models_pruned'
