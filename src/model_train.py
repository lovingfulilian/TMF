"""
model_train - 训练模型

Author: 骆昊
Version: 0.0.1
"""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

from src.config import Config

# 加载数据
df = pd.read_csv(Config.train_pre_file)
X_train, y_train = df.text, df.label
df = pd.read_csv(Config.valid_pre_file)
X_valid, y_valid = df.text, df.label

# 构造流水线
pl = Pipeline(steps=[
    ('vec', TfidfVectorizer(ngram_range=(1, 2), min_df=0.0001, max_df=0.99)),
    ('sel', SelectKBest(k=8192)),
    ('clf', RandomForestClassifier(n_estimators=100)),
])
# 喂入数据
pl.fit(X_train, y_train)
# 验证效果
y_pred = pl.predict(X_valid)

# 模型评估
print(confusion_matrix(y_valid, y_pred))
print(classification_report(y_valid, y_pred))

# 保存模型（序列化）
joblib.dump(pl, Config.model_file)
