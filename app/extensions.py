"""
extensions - 扩展功能

Author: 骆昊
Version: 0.0.1
"""
import joblib


class TextClassifierExtension:
    """文本分类器扩展"""

    def __init__(self):
        self.text_clf_model = None

    def init_app(self, model_path: str):
        print('===== [extensions] 正在加载文本分类模型 =====')
        if not self.text_clf_model:
            self.text_clf_model = joblib.load(model_path)
        print('===== [extensions] 文本分类模型加载完成 =====')


thy_extension = TextClassifierExtension()
