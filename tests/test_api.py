"""
test_api - API 接口测试

Author: 骆昊
Version: 0.0.1
"""
import requests

BASE_URL = "http://127.0.0.1:5000"


def test_base_url():
    """测试基路径"""
    resp = requests.get(BASE_URL)
    assert resp.status_code == 200
