"""
model_eval - 模型评估

Author: 骆昊
Version: 0.0.1
"""
import joblib
from sklearn.pipeline import Pipeline

from config import Config


def main():
    pl = joblib.load(Config.model_file)  # type: Pipeline
    # Todo: 完成模型推理（使用甲方测试数据对模型做验证）


if __name__ == '__main__':
    main()
