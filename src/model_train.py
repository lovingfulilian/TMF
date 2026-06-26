"""
model_train - 训练模型

Author: 骆昊
Version: 0.0.1
"""
import fasttext
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

from src.config import Config

def train_model():
    """模型训练"""
    # clf = fasttext.train_supervised(
    #     input=str(Config.train_pre_file),
    #     lr=0.5,         # 学习率
    #     epoch=32,       # 训练轮次
    #     dim=128,        # 特征维度
    #     minCount=3,     # 最低词频
    #     wordNgrams=2,   # N元词袋
    #     loss='hs'       # 损失函数 - 层次 Softmax
    # )
    #
    # clf.quantize(
    #     input=str(Config.valid_pre_file),
    #     cutoff=100000,  # 保留特征数量
    #     retrain=True,   # 开启微调
    #     lr=0.1,         # 学习率
    #     epoch=4,        # 训练轮次
    # )

    clf = fasttext.train_supervised(
        input=str(Config.train_pre_file),
        autotuneValidationFile=str(Config.valid_pre_file),
        autotuneMetric='f1',
        autotuneModelSize='50M',
        autotuneDuration=300
    )

    clf.save_model(str(Config.ftz_model_file))
