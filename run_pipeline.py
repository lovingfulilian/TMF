"""
run_pipeline - 离线训练流水线

Author: 骆昊
Version: 0.0.1
"""
from src.data_pre import clean_data
from src.model_eval import evaluate_model
from src.model_train import train_model
from src.model_compress import distill_model, evaluate_distilled_model, quantize_model, evaluate_quantized_model


def main():
    print("\n=== [Step 1/5] 开始执行数据清洗和准备工作 ===")
    # clean_data()

    print("\n=== [Step 2/5] 开始训练和导出文本分类模型 ===")
    # train_model()

    print("\n=== [Step 3/5] 对已有模型进行量化或者蒸馏 ===")
    distill_model()

    print("\n=== [Step 4/5] 对量化或蒸馏的模型进行评估 ===")
    evaluate_distilled_model()

    print("\n=== [Step 5/5] 启动服务之前对模型进行评估 ===")
    evaluate_model()


if __name__ == "__main__":
    main()
