"""
predict - API 接口实现

Author: 骆昊
Version: 0.0.1
"""
from flask import Blueprint

# 创建蓝图对象
model_bp = Blueprint('models', __name__, url_prefix='/api/v1/models')


@model_bp.route('/clf', methods=['GET', 'POST'])
def text_clf_predict():
    return {'code': 200, 'result': 'wonderful'}
