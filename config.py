"""
项目全局配置
所有模块共享的路径与常量定义，避免硬编码
"""
from pathlib import Path

# ===== 路径配置 =====
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # 原始数据(成员1输出)
PROCESSED_DIR = DATA_DIR / "processed"  # 清洗后数据(成员2输出)

MODEL_DIR = PROJECT_ROOT / "model"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"

RESULTS_DIR = PROJECT_ROOT / "results"
AGENT_DIR = PROJECT_ROOT / "agent"
CRAWLER_DIR = PROJECT_ROOT / "crawler"

# ===== 数据集文件 =====
TRAIN_FILE = DATA_DIR / "train.jsonl"
VAL_FILE = DATA_DIR / "val.jsonl"
TEST_FILE = DATA_DIR / "test.jsonl"
LABELS_FILE = DATA_DIR / "labels.json"

# ===== 分类映射 =====
CATEGORY_MAPPING_FILE = PROJECT_ROOT / "category_mapping.json"

# ===== 采集配置(成员1) =====
CRAWLER_CONFIG = {
    "target_total": 3000,          # 目标文献总量
    "min_categories": 100,         # 最少类别数
    "request_interval": (4, 8),     # 请求随机间隔(秒)
    "max_retries": 3,              # 单条重试次数
    "headless": False,              # CNKI 检测 headless Chrome → 跳验证码, 必须 False
    "campus_network": True,        # 校园网环境(无需代理)
}

# ===== 模型配置(成员3) =====
MODEL_CONFIG = {
    "max_length": 512,
    "batch_size": 8,
    "epochs": 20,
    "learning_rate": 2e-5,
    "baseline_model": "tfidf_svm",          # 基线模型
    "deep_model": "medbert_v20",             # 默认部署模型
}

# ===== Agent配置(成员4) =====
AGENT_CONFIG = {
    "llm_model": "gpt-4o-mini",
    "top_k": 5,                    # Top-K候选类别数
    "confidence_threshold": 0.6,   # 低置信度阈值
}

# ===== Web配置(成员5) =====
WEB_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
}


def ensure_dirs():
    """确保所有必需目录存在"""
    for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, MODEL_DIR,
              CHECKPOINT_DIR, RESULTS_DIR, AGENT_DIR, CRAWLER_DIR]:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"已创建所有必需目录")
