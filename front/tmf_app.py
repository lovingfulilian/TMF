"""
tmf_app - Streamlit 用户界面

日志级别：DEBUG < INFO < WARNING < ERROR < CRITICAL

Author: 骆昊
Version: 0.0.1
"""
import sys
from pathlib import Path

import requests
import streamlit as st
from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent
BASE_URL = "http://127.0.0.1:8080"

logger.remove()
logger.add(sys.stderr, level='WARNING')
logger.add(
    BASE_DIR / 'logs/api_service.log',
    rotation='100 MB',   # 何时关闭当前日志文件并创建新文件
    retention='7 days',  # 旧日志文件的回收与清理策略
    serialize=True,      # 是否将日志转换为 JSON 字符串输出
    level='INFO',        # 日志处理器接收的最低日志级别
    enqueue=True,        # 日志打印非阻塞
    catch=True,          # 防止日志写入失败导致应用崩溃
    compression='zip',   # 当日志触发切分时旧的日志文件会被自动压缩
)


def get_class_label(text):
    """获取类别标签"""
    try:
        resp = requests.get(
            url=f'{BASE_URL}/api/v2/predict',
            headers={'content-type': 'application/json'},
            json={'text': text},
        )
        result = resp.json()
        if result['code'] == 0:
            return result['label']
        return result['message']
    except Exception as e:
        logger.error(str(e))
        return ''


st.write('## 投满分（文本分类专家）')
content = st.text_input(label='请输入要分类的文本内容：')
ok_button = st.button('确定')

if ok_button and content.strip():
    class_label = get_class_label(content)
    if class_label:
        st.write(f'**分类结果**：{class_label}')
    else:
        st.error('服务器维护中，请稍后再尝试访问')
