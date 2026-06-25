"""
model_eval - 模型评估

Author: 骆昊
Version: 0.0.1
"""
import time

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from src.config import Config


def evaluate_model():
    df = pd.read_csv(Config.test_pre_file)
    pl = joblib.load(Config.pkl_model_file)  # type: Pipeline

    total_accuracy = 0.0
    total_duration = 0.0
    for _ in range(100):
        temp = df.sample(n=10)
        X_test, y_test = temp.text.values, temp.label.values
        start = time.perf_counter()
        y_pred = pl.predict(X_test)
        end = time.perf_counter()
        total_accuracy += accuracy_score(y_test, y_pred)
        total_duration += end - start

    print(f'Accuracy: {total_accuracy / 100:.2%}')
    print(f'Duration: {total_duration / 100:.3f}')
