import re
import unicodedata
from html.parser import HTMLParser
from typing import Protocol

from agent.schemas import PaperFields, ParsedDocument, PreparedText, WarningInfo


class Preprocessor(Protocol):
    version: str

    def prepare(self, document: ParsedDocument) -> PreparedText: ...


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1
        elif tag in {"p", "br", "div", "li", "tr"} and not self.hidden:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)
        elif tag in {"p", "div", "li", "tr"} and not self.hidden:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def clean_text(text: str) -> str:
    """Default demo rule. A real model must use this version or inject member 2's rule."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Do not mistake mathematical comparisons such as CD4<200 for HTML tags.
    if re.search(r"</?(?:html|body|p|br|div|span|script|style|table|b|i|a)\b[^>]*>", text, re.I):
        parser = _TextExtractor()
        parser.feed(text)
        parser.close()
        text = "".join(parser.parts)
    text = "".join(
        ch
        for ch in text
        if (unicodedata.category(ch) != "Cc" or ch in "\n\t") and ch not in "\u200b\ufeff"
    )
    return "\n".join(
        line for raw in text.splitlines() if (line := re.sub(r"[^\S\n]+", " ", raw).strip())
    ).strip()


def build_model_input(fields: PaperFields) -> str:
    """The field order and separators form part of preprocess_version."""
    parts = []
    for name, value in (
        ("标题", fields.title),
        ("关键词", "；".join(fields.keywords)),
        ("摘要", fields.abstract),
    ):
        value = clean_text(value)
        if value:
            parts.append(f"{name}：{value}")
    return "\n".join(parts)


def extract_paper_fields(text: str) -> PaperFields | None:
    """Conservative heading-based extraction; no claim of layout/semantic recognition."""
    title = ""
    keywords: list[str] = []
    abstract: list[str] = []
    in_abstract = False
    for line in text.splitlines():
        if match := re.match(r"^(?:标题|Title)\s*[:：]\s*(.+)$", line, re.I):
            title = match[1]
            in_abstract = False
        elif match := re.match(r"^(?:关键词|关键字|Keywords?)\s*[:：]\s*(.*)$", line, re.I):
            keywords = [part.strip() for part in re.split(r"[;；,，、]", match[1]) if part.strip()]
            in_abstract = False
        elif match := re.match(r"^(?:摘要|Abstract)(?:\s*[:：]\s*|\s*$)(.*)$", line, re.I):
            abstract.append(match[1])
            in_abstract = True
        elif re.match(
            r"^(?:\d+(?:\.\d+)*[.、\s]+|引言|前言|参考文献|正文|Introduction\b|References\b)",
            line,
            re.I,
        ):
            in_abstract = False
        elif in_abstract:
            abstract.append(line)
    if not any(part.strip() for part in abstract):
        return None
    return PaperFields(title=title, keywords=keywords, abstract="\n".join(abstract).strip())


class DefaultTextPreprocessor:
    version = "builtin-v1"

    def prepare(self, document: ParsedDocument) -> PreparedText:
        if document.fields is not None:
            return PreparedText(
                text=build_model_input(document.fields), strategy="structured_fields"
            )
        cleaned = clean_text(document.text)
        if document.source_type == "text":
            return PreparedText(text=cleaned, strategy="free_text")
        fields = extract_paper_fields(cleaned)
        if fields is not None:
            warnings = []
            if not fields.title or not fields.keywords:
                warnings.append(
                    WarningInfo(
                        code="PARTIAL_PAPER_FIELDS",
                        message="已提取摘要，部分标题或关键词字段未识别。",
                    )
                )
            return PreparedText(
                text=build_model_input(fields), strategy="document_fields", warnings=warnings
            )
        return PreparedText(
            text=cleaned,
            strategy="document_fulltext",
            warnings=[
                WarningInfo(
                    code="FULLTEXT_FALLBACK",
                    message="未识别到明确的摘要字段，已使用文档正文进行分类。",
                )
            ],
        )
