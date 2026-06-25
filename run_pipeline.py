"""
run_pipeline - 离线训练流水线

Author: 骆昊
Version: 0.0.1
"""
import sys
from pathlib import Path

from src.model_eval import evaluate_model
# from src.model_train import model_train
# from src.data_pre import clean_data

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))


def main():
    """离线训练流水线程序入口"""
    print("\n=== [Step 1/3] 开始执行数据清洗和准备工作 ===")
    # clean_data()

    print("\n=== [Step 2/3] 开始训练和导出文本分类模型 ===")
    # model_train()

    print("\n=== [Step 3/3] 启动服务之前对模型进行评估 ===")
    evaluate_model()


if __name__ == "__main__":
    main()
