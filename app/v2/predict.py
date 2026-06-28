"""
predict - API 接口实现

Author: 骆昊
Version: 0.0.1
"""
from flask import Blueprint, request
from app.extensions import thy_extension
from src.data_pre import cut_zh_words

# 创建蓝图对象
model_bp_v2 = Blueprint('models', __name__, url_prefix='/api/v2')


# 路由 - 用户请求过来之后根据URL决定调用哪个函数
@model_bp_v2.route('/predict', methods=['GET', 'POST'])
def text_clf_predict():
    """文本分类预测"""
    # HTTP 请求 - 请求行 / 请求头 / 空行 / 消息体
    # Header ---> content-type: application/json
    # silent=True - 如果解析 JSON 失败不抛出异常
    # force=True - 强制将消息体解析为 JSON 格式
    json_obj = request.get_json(silent=True, force=True)
    if not json_obj:
        return {'code': -20, 'message': '请求数据不符合 JSON 格式规范'}

    text = json_obj.get('text', '')
    if not text:
        return {'code': -10, 'message': '请提供要分类的文本内容'}

    text = cut_zh_words(text)
    y_pred, _ = thy_extension.ftz_clf_model.predict([text])
    label_index = int(y_pred[0][0][9:])
    return {'code': 0, 'message': 'OK', 'label': thy_extension.class_labels[label_index]}
