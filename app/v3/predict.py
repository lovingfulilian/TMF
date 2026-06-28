"""
predict - API 接口实现

Author: 骆昊
Version: 0.0.1
"""
import torch
from flask import Blueprint, request

from app.extensions import thy_extension

# 创建蓝图对象
model_bp_v3 = Blueprint('models', __name__, url_prefix='/api/v3')


# 路由 - 用户请求过来之后根据URL决定调用哪个函数
@model_bp_v3.route('/predict', methods=['GET', 'POST'])
def text_clf_predict():
    """文本分类预测"""
    json_string = request.get_json(silent=True, force=True)
    if not json_string:
        return {'code': -20, 'message': '请求数据不符合 JSON 格式规范'}

    text = json_string.get('text', '')
    if not text:
        return {'code': -10, 'message': '请提供要分类的文本内容'}

    inputs = thy_extension.tokenizer(
        [text],
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=32
    )
    input_ids = inputs.input_ids.to(thy_extension.device)
    attention_mask = inputs.attention_mask.to(thy_extension.device)
    with torch.inference_mode():
        output = thy_extension.macbert_model(input_ids=input_ids, attention_mask=attention_mask)
    y_pred = torch.argmax(output.logits, dim=-1).cpu().numpy()

    return {'code': 0, 'message': 'OK', 'label': thy_extension.class_labels[y_pred[0]]}
