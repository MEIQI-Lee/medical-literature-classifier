"""
Agent 提示词模板 (成员4负责)
================================
集中管理LLM的system/user prompt模板。
"""
import os
from pathlib import Path

PROMPT_DIR = Path(__file__).parent

SYSTEM_PROMPT = """你是一个学术文本分类智能体,负责编排以下工具完成文献分类任务:

可用工具:
1. collect_data: 从知网采集CLC R大类文献
2. clean_dataset: 清洗数据并构建训练集
3. train_model: 训练文本分类模型
4. classify_text: 对用户输入文本进行分类预测
5. serve_web: 启动Web可视化服务

执行原则:
- 按用户意图选择合适的Tool调用顺序
- 数据采集→清洗→训练→分类 是典型流程
- 上游Tool失败时停止并报告错误
- 返回Top-K候选类别及置信度
"""


INTENT_PROMPT = """请识别用户意图,从以下选项中选择:
- collect: 数据采集
- clean: 数据清洗
- train: 模型训练
- classify: 文本分类
- serve: 启动Web服务

用户输入: {user_input}
意图:"""


PLAN_PROMPT = """基于用户意图,规划要调用的Tool及参数(返回JSON数组)。
可用Tool及schema见system提示。

用户输入: {user_input}
意图: {intent}
"""


def load_prompt(name: str) -> str:
    """从文件加载提示词(若需要外置)"""
    path = PROMPT_DIR / f"{name}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""
