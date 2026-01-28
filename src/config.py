"""
Typst Coder 项目配置文件 / Typst Coder project configuration file
"""

# 数据集下载链接 / Dataset download URLs
DATASET_URLS = {
    "train": "https://huggingface.co/datasets/TechxGenus/Typst-Train/resolve/main/typst_train.json?download=true",
    "test": "https://huggingface.co/datasets/TechxGenus/Typst-Test/resolve/main/typst_test.json?download=true"
}

# 模型配置 / Model configuration
MODEL_CONFIG = {
    "name": "Qwen/Qwen3-0.6B",
    "local_path": "model/qwen3-0.6b",
    "trust_remote_code": True
}

# 训练配置 / Training configuration
TRAINING_CONFIG = {
    # 基础参数 / Basic parameters
    "model_name_or_path": "model/qwen3-0.6b",
    "data_dir": "data/processed",
    "output_dir": "outputs/qwen3-0.6b-typst",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 1e-4,
    "max_seq_length": 1024,

    # 优化器参数 / Optimizer parameters
    "weight_decay": 0.01,
    "warmup_steps": 500,
    "lr_scheduler_type": "cosine",

    # 混合精度和分布式 / Mixed precision and distributed
    "fp16": True,
    "ddp_find_unused_parameters": False,

    # 日志和保存 / Logging and saving
    "logging_steps": 10,
    "save_steps": 500,
    "eval_steps": 500,
    "save_total_limit": 3,
    "do_train": True,
    "do_eval": True,

    # Lora配置 / Lora configuration
    "use_lora": True,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
}

# 推理配置 / Inference configuration
INFERENCE_CONFIG = {
    "model_path": "outputs/qwen3-0.6b-typst/checkpoint-*/",
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1
}
