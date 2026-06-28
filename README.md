# 项目说明

作者：骆昊

## 注意事项

### 克隆项目

```bash
git clone https://gitee.com/jackfrued/tmf_v1.git
```

### 还原环境

Conda:

```bash
conda env create -f https://gitee.com/jackfrued/tmf_v1.git
```

Pip:

```bash
pip install -r requirements.txt
```

### 测试用例

```bash
pytest tests/ -v
pytest tests/ -v -m api
```
