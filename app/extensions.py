"""
extensions - 扩展功能

Author: 骆昊
Version: 0.0.1
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.config import Config


class TextClassifierExtension:
    """文本分类器扩展"""

    def __init__(self):
        self.macbert_model = None
        self.tokenizer = None
        self.class_labels = open(Config.class_file, encoding='utf-8').read().strip().splitlines()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available() else
            'cpu'
        )

    def init_app(self, model_path: str):
        print('===== [extensions] 正在加载文本分类模型 =====')
        if not self.macbert_model:
            self.tokenizer = AutoTokenizer.from_pretrained(Config.model_output_dir)
            self.macbert_model = AutoModelForSequenceClassification.from_pretrained(Config.model_output_dir)
            self.macbert_model.to(self.device)
        print('===== [extensions] 文本分类模型加载完成 =====')


thy_extension = TextClassifierExtension()
