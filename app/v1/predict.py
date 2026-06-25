"""
predict - API 接口实现

Author: 骆昊
Version: 0.0.1
"""
from flask import Blueprint, g
from app.extensions import thy_extension

# 创建蓝图对象
model_bp = Blueprint('models', __name__, url_prefix='/api/v1/models')


@model_bp.before_request
def before_request():
    """每个请求进来时把启动时加载好的模型单例安全地绑定到当前请求的 g 对象上"""
    g.model = thy_extension.text_clf_model


@model_bp.route('/clf', methods=['GET', 'POST'])
def text_clf_predict():
    print(g.model)
    return {'code': 200, 'result': 'wonderful'}
