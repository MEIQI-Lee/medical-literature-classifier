from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, revalidate_instances="always")


class WarningInfo(Schema):
    code: str
    message: str


class ErrorInfo(Schema):
    code: str
    message: str
    step: str | None = None


class StepInfo(Schema):
    request_id: str
    sequence: int
    step: str
    status: Literal["running", "success", "error"]
    duration_ms: float = 0
    message: str


class ModelInfo(Schema):
    model_version: str = Field(min_length=1)
    label_version: str = Field(min_length=1)
    preprocess_version: str = Field(min_length=1)
    label_ids: list[Annotated[int, Field(strict=True, ge=0)]] = Field(min_length=2)
    score_type: Literal["probability", "decision_score"]
    calibrated: bool = False
    mode: Literal["model", "demo"] = "model"

    @model_validator(mode="after")
    def validate_labels(self):
        if any(type(label) is not int or label < 0 for label in self.label_ids):
            raise ValueError("label_ids must contain nonnegative integers")
        if len(set(self.label_ids)) != len(self.label_ids):
            raise ValueError("label_ids must be unique")
        if self.calibrated and self.score_type != "probability":
            raise ValueError("calibrated outputs must be probabilities")
        return self


class ModelScore(Schema):
    label_id: int = Field(strict=True, ge=0)
    score: float


class ModelOutput(Schema):
    model_info: ModelInfo
    predictions: list[ModelScore] = Field(min_length=1)
    truncated: bool = False
    input_tokens: int | None = Field(default=None, ge=0)


class ConfidencePolicy(Schema):
    # Thresholds are tied to a model; no production threshold is guessed by default.
    model_version: str
    min_probability: float = Field(ge=0, le=1)
    min_margin: float = Field(ge=0, le=1)


class AgentConfig(Schema):
    top_k: int = Field(default=5, strict=True, ge=1)
    min_text_chars: int = Field(default=10, strict=True, ge=1)
    max_text_chars: int = Field(default=200_000, strict=True, ge=1)
    max_file_bytes: int = Field(default=10 * 1024 * 1024, strict=True, ge=1)
    max_pdf_pages: int = Field(default=100, strict=True, ge=1)
    max_docx_uncompressed_bytes: int = Field(default=50 * 1024 * 1024, strict=True, ge=1)
    max_docx_entries: int = Field(default=2000, strict=True, ge=1)
    confidence: ConfidencePolicy | None = None

    @model_validator(mode="after")
    def validate_lengths(self):
        if self.min_text_chars > self.max_text_chars:
            raise ValueError("min_text_chars exceeds max_text_chars")
        return self


class PaperFields(Schema):
    title: str = ""
    keywords: list[str] = Field(default_factory=list)
    abstract: str = ""


class ParsedDocument(Schema):
    text: str
    source_type: Literal["text", "paper", "txt", "pdf", "docx"]
    fields: PaperFields | None = None
    filename: str | None = None
    pages: int | None = None
    warnings: list[WarningInfo] = Field(default_factory=list)


class PreparedText(Schema):
    text: str
    strategy: Literal["free_text", "structured_fields", "document_fields", "document_fulltext"]
    warnings: list[WarningInfo] = Field(default_factory=list)


class Category(Schema):
    code: str = Field(min_length=1)
    name: str = Field(min_length=1)
    path: list[str] = Field(default_factory=list)


class CategoryMapping(Schema):
    version: str
    categories: list[Category] = Field(min_length=1)


class LabelEntry(Schema):
    label_id: int = Field(strict=True, ge=0)
    category_code: str = Field(min_length=1)


class LabelMapping(Schema):
    version: str
    category_version: str
    labels: list[LabelEntry] = Field(min_length=2)


class Candidate(Schema):
    rank: int
    label_id: int
    category_code: str
    category_name: str
    category_path: list[str]
    score: float
    # A decision score is not a probability and must not be formatted as a percentage.
    confidence: float | None


class InputSummary(Schema):
    source_type: str
    filename: str | None = None
    pages: int | None = None
    extracted_chars: int = 0
    prepared_chars: int = 0
    preparation_strategy: str | None = None
    truncated: bool = False
    input_tokens: int | None = None


class AgentResult(Schema):
    schema_version: str = "1.0"
    request_id: str
    status: Literal["success", "error"]
    mode: Literal["model", "demo"]
    prediction: Candidate | None = None
    candidates: list[Candidate] = Field(default_factory=list)
    confidence_status: Literal["normal", "low", "unknown"] = "unknown"
    confidence_threshold: float | None = None
    requested_top_k: int = 5
    returned_top_k: int = 0
    display_mode: Literal["top1", "candidates", "error"]
    message: str
    model_info: ModelInfo
    input: InputSummary
    steps: list[StepInfo]
    warnings: list[WarningInfo] = Field(default_factory=list)
    duration_ms: float
    error: ErrorInfo | None = None
