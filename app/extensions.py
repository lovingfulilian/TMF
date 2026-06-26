"""
extensions - 扩展功能

Author: 骆昊
Version: 0.0.1
"""
import fasttext
# import joblib

from src.config import Config


class TextClassifierExtension:
    """文本分类器扩展"""

    def __init__(self):
        # self.text_clf_model = None
        self.ftz_clf_model = None
        self.class_labels = None

    def init_app(self, model_path: str):
        print('===== [extensions] 正在加载文本分类模型 =====')
        self.class_labels = open(Config.class_file, encoding='utf-8').read().strip().splitlines()
        # if not self.text_clf_model:
        #     self.text_clf_model = joblib.load(model_path)
        if not self.ftz_clf_model:
            self.ftz_clf_model = fasttext.load_model(str(Config.ftz_model_file))
        print('===== [extensions] 文本分类模型加载完成 =====')


thy_extension = TextClassifierExtension()
