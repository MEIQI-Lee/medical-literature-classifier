"""Use the same field builder as the training dataset, exactly once."""

import hashlib
import re
from pathlib import Path

from agent.schemas import PaperFields, ParsedDocument, PreparedText, WarningInfo
from agent.tools.text import extract_paper_fields
from preprocess import clean

PREPROCESS_VERSION = (
    "member2-fields-v1-" + hashlib.sha256(Path(clean.__file__).read_bytes()).hexdigest()[:12]
)


def _explicit_fields(text: str) -> PaperFields | None:
    # Also recognize a serialized dataset row: 标题：…。关键词：…。摘要：…
    headings = list(
        re.finditer(
            r"(?:^|[。\n\r])\s*(标题|关键词|关键字|摘要|Title|Keywords?|Abstract)\s*[:：]\s*",
            text,
            re.IGNORECASE,
        )
    )
    if not headings:
        return extract_paper_fields(text)
    values = {"title": "", "abstract": "", "keywords": []}
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        value = text[heading.end() : end].strip()
        name = heading[1].lower()
        if name in {"标题", "title"}:
            values["title"] = value
        elif name in {"摘要", "abstract"}:
            # Stop at conventional body/reference headings in extracted documents.
            value = re.split(
                r"\n\s*(?:引言|前言|正文|参考文献|Introduction\b|References\b|\d+[.、\s])",
                value,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
            values["abstract"] = value
        else:
            values["keywords"] = clean.parse_keywords(value.replace("、", "；"))
    return PaperFields(**values)


class TeamPreprocessor:
    version = PREPROCESS_VERSION

    def prepare(self, document: ParsedDocument) -> PreparedText:
        if document.fields is not None:
            fields, strategy = document.fields, "structured_fields"
        else:
            # Preserve newlines until heading extraction has finished.
            text = document.text.replace("\r\n", "\n").replace("\r", "\n")
            fields = _explicit_fields(text)
            strategy = "structured_fields" if document.source_type == "text" else "document_fields"
        if fields is not None:
            warnings = []
            if not fields.title or not fields.abstract or not fields.keywords:
                warnings.append(
                    WarningInfo(
                        code="PARTIAL_PAPER_FIELDS",
                        message="部分标题、摘要或关键词字段未提供或未识别。",
                    )
                )
            return PreparedText(
                text=clean.build_text_input(fields.title, fields.keywords, fields.abstract),
                strategy=strategy,
                warnings=warnings,
            )
        is_document = document.source_type != "text"
        return PreparedText(
            text=clean.clean_special_chars(document.text),
            strategy="document_fulltext" if is_document else "free_text",
            warnings=[
                WarningInfo(
                    code="FULLTEXT_FALLBACK",
                    message="未识别到明确的论文字段，已使用文档正文进行分类。",
                )
            ]
            if is_document
            else [],
        )
