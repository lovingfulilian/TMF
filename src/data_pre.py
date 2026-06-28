"""
data_pre - 数据预处理

Author: 骆昊
Version: 0.0.1
"""
import re

from src.config import Config

# 清洗文本的正则表达式
SPACE_PATTERN = re.compile(r'[\s\u3000]+')
BRACE_PATTERN = re.compile(r'\[.*?\]|\(.*?\)|【.*?】|（.*?）|「.*?」')
ATUSR_PATTERN = re.compile(r'@\w+')
TOPIC_PATTERN = re.compile(r'#.*?#')
MYURL_PATTERN = re.compile(r'https?://[^\s\u4e00-\u9fa5]+')
CHENN_PATTERN = re.compile(r'[^a-zA-Z0-9\s\u4e00-\u9fa5:,.?!;：，。？！；]+')
NOENN_PATTERN = re.compile(r'[^\s\u4e00-\u9fa5：，。？！；]+')

# 加载停用词表
with open(Config.stopwords_file, encoding='utf-8') as file_object:
    STOP_WORDS = set(file_object.read().splitlines())


def clean_raw_text(text: str, allow_eng_num: bool=True) -> str:
    """清理原始文本内容"""
    text = re.sub(SPACE_PATTERN, ' ', text)
    text = re.sub(BRACE_PATTERN, '', text)
    text = re.sub(ATUSR_PATTERN, '', text)
    text = re.sub(TOPIC_PATTERN, '', text)
    text = re.sub(MYURL_PATTERN, '', text)

    if allow_eng_num:
        text = re.sub(CHENN_PATTERN, '', text)
    else:
        text = re.sub(NOENN_PATTERN, '', text)

    text = re.sub(SPACE_PATTERN, ' ', text)
    return text.strip()


def get_corpus(corpus_file):
    """获取指定文件中的语料"""
    corpus = []
    with open(corpus_file, encoding='utf-8') as file_obj:
        content = file_obj.read()
    content = re.sub(r'\t+', '\t', content)
    for line in content.splitlines():
        doc, label = line.split('\t', maxsplit=1)
        corpus.append((doc, int(label)))
    return corpus


def clean_data():
    """清洗数据"""
    pass
