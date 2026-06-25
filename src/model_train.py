"""
model_train - 训练模型

Author: 骆昊
Version: 0.0.1
"""
import joblib
import pandas as pd
# from skl2onnx import to_onnx
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

from src.config import Config

def train_model():
    """模型训练"""
    # 加载数据
    df = pd.read_csv(Config.train_pre_file)
    X_train, y_train = df.text.values, df.label.values
    df = pd.read_csv(Config.valid_pre_file)
    X_valid, y_valid = df.text.values, df.label.values

    # 随机森林模型参数（控制预剪枝）
    rf_params = {
        'n_estimators': 128,
        'min_samples_split': 16,
        'max_features': 'log2',
        'n_jobs': -1
    }

    # 构造流水线
    pl = Pipeline(steps=[
        # TfidfVectorizer ---> 将分词后的中文文本处理成稀疏向量（矩阵）- SparseMatrix
        ('vec', TfidfVectorizer(ngram_range=(1, 2), min_df=0.0001, max_df=0.99)),
        # SelectKBest ---> 特征选择 ---> 方差分析（最大化组间方差最小化组内方差）
        ('sel', SelectKBest(k=8192)),
        # RandomForestClassifier ---> 集成学习（Bagging）
        ('clf', RandomForestClassifier(**rf_params)),
    ])
    # 喂入数据
    pl.fit(X_train, y_train)
    # 验证效果
    y_pred = pl.predict(X_valid)

    # 模型评估 ---> 分类模型 ---> Accuracy / Precision / Recall / F1 Score / AUC
    print(confusion_matrix(y_valid, y_pred))
    print(classification_report(y_valid, y_pred))

    # 保存模型（序列化）
    joblib.dump(pl, Config.pkl_model_file, compress=3)

    # # 将模型保存为 ONNX - Open Neural Network eXchange
    # # 需要安装 skl2onnx - conda install skl2onnx -c conda-forge
    # initial_type = [('text_input', ['str', None, 1])]
    # onnx_pipeline = to_onnx(pl, initial_types=initial_type)
    # with open(Config.onnx_model_file, 'wb') as file_obj:
    #     file_obj.write(onnx_pipeline.SerializeToString())
