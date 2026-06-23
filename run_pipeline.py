"""
run_pipeline - 离线训练流水线

Author: 骆昊
Version: 0.0.1
"""
import sys
from pathlib import Path

# 将项目根目录加入 Python 搜索路径
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

# from src.data_clean import clean_raw_data
# from src.model_train import train_and_export_model


def main():
    """离线训练流水线程序入口"""
    print("\n=== [Step 1/3] 开始执行数据清洗和准备工作 ===")
    # Todo: 数据清洗

    print("\n=== [Step 2/3] 开始训练和导出文本分类模型 ===")
    # Todo: 训练导出

    print("\n=== [Step 3/3] 启动服务之前对模型进行评估 ===")
    # Todo: 模型评估


if __name__ == "__main__":
    main()
