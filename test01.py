"""
test01 - Flask 快速上手

Author: 骆昊
Version: 0.0.1
"""
import joblib
from flask import Flask, request
from sklearn.pipeline import Pipeline

from src.config import Config
from src.data_pre import cut_zh_words

app = Flask(__name__)
pl = joblib.load(Config.pkl_model_file)  # type: Pipeline
class_labels = open(Config.class_file, encoding='utf-8').read().strip().splitlines()


@app.route('/')
def hello_world():
    return '<h1>Hello, World!</h1>'


@app.route('/api/predict', methods=['GET', 'POST'])
def text_classify():
    json_string = request.get_json()
    text = json_string.get('text', '')
    if text:
        text = cut_zh_words(text)
        y_pred = pl.predict([text])
        return {'label': class_labels[y_pred[0]], 'code': 0, 'message': 'OK'}
    return {'code': -10, 'message': '请提供要分类的文本内容'}


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080, debug=True)
