import json
from pathlib import Path

from pydantic import ValidationError

from agent.errors import AgentError
from agent.schemas import Category, CategoryMapping, LabelMapping, ModelInfo


class CategoryCatalog:
    def __init__(self, labels: LabelMapping, categories: CategoryMapping):
        self.label_version = labels.version
        self.category_version = categories.version
        self._codes = {item.label_id: item.category_code for item in labels.labels}
        self._categories = {item.code: item for item in categories.categories}
        if len(self._codes) != len(labels.labels):
            raise AgentError("LABEL_MAPPING_INVALID", "标签表中存在重复的 label_id。")
        if len(set(self._codes.values())) != len(self._codes):
            raise AgentError(
                "LABEL_MAPPING_INVALID", "单标签任务中不同 label_id 不能重复指向同一类别。"
            )
        if len(self._categories) != len(categories.categories):
            raise AgentError("LABEL_MAPPING_INVALID", "类别表中存在重复编码。")
        if labels.category_version != categories.version:
            raise AgentError("LABEL_VERSION_MISMATCH", "标签表与类别表的版本不匹配。")
        if not set(self._codes.values()).issubset(self._categories):
            raise AgentError("LABEL_MAPPING_INVALID", "标签表引用了不存在的类别编码。")

    @classmethod
    def from_files(cls, labels_path: str | Path, categories_path: str | Path):
        try:
            labels = LabelMapping.model_validate_json(Path(labels_path).read_text(encoding="utf-8"))
            categories = CategoryMapping.model_validate_json(
                Path(categories_path).read_text(encoding="utf-8")
            )
        except (OSError, ValueError, ValidationError, json.JSONDecodeError) as exc:
            raise AgentError(
                "LABEL_MAPPING_INVALID", "无法读取类别或标签映射，请检查文件格式。"
            ) from exc
        return cls(labels, categories)

    def validate_model(self, model: ModelInfo):
        if model.label_version != self.label_version:
            raise AgentError("LABEL_VERSION_MISMATCH", "模型与标签表版本不匹配。")
        if set(model.label_ids) != set(self._codes):
            raise AgentError("LABEL_MAPPING_INVALID", "模型声明的标签集合与标签表不一致。")

    def query(self, label_id: int) -> Category:
        try:
            return self._categories[self._codes[label_id]].model_copy(deep=True)
        except KeyError as exc:
            raise AgentError("UNKNOWN_LABEL", "模型返回了标签表中不存在的类别。") from exc
