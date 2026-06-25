"""
run_pipeline - 离线训练流水线

Author: 骆昊
Version: 0.0.1
"""
from src.model_eval import evaluate_model
from src.model_train import train_model
from src.data_pre import clean_data


def main():
    """离线训练流水线程序入口"""
    print("\n=== [Step 1/3] 开始执行数据清洗和准备工作 ===")
    clean_data()

    print("\n=== [Step 2/3] 开始训练和导出文本分类模型 ===")
    train_model()

    print("\n=== [Step 3/3] 启动服务之前对模型进行评估 ===")
    evaluate_model()


if __name__ == "__main__":
    main()
