"""
predict - API 接口实现

Author: 骆昊
Version: 0.0.1
"""
from flask import Blueprint, g, request
from app.extensions import thy_extension
from src.data_pre import cut_zh_words

# 创建蓝图对象
model_bp = Blueprint('models', __name__, url_prefix='/api/v1')


@model_bp.before_request
def before_request():
    """每个请求进来时把启动时加载好的模型单例安全地绑定到当前请求的 g 对象上"""
    g.model = thy_extension.text_clf_model
    g.class_labels = thy_extension.class_labels


@model_bp.route('/predict', methods=['GET', 'POST'])
def text_clf_predict():
    """文本分类预测"""
    json_string = request.get_json()
    text = json_string.get('text', '')
    if text:
        text = cut_zh_words(text)
        y_pred = g.model.predict([text])
        return {'label': g.class_labels[y_pred[0]], 'code': 0, 'message': 'OK'}
    return {'code': -10, 'message': '请提供要分类的文本内容'}
