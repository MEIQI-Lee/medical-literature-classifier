import io
import zipfile
from pathlib import PurePosixPath

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

from agent.errors import AgentError
from agent.schemas import AgentConfig, ParsedDocument, WarningInfo


def _decode_text(content: bytes) -> tuple[str, list[WarningInfo]]:
    if content.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return content.decode("utf-16"), []
        except UnicodeError as exc:
            raise AgentError("TEXT_ENCODING_ERROR", "TXT 文件的 UTF-16 编码不完整。") from exc
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            text = content.decode(encoding)
            if "\x00" in text:
                raise AgentError("INVALID_FILE", "文件包含二进制内容，请上传有效 TXT 文件。")
            notes = (
                []
                if encoding == "utf-8-sig"
                else [
                    WarningInfo(
                        code="ENCODING_FALLBACK", message="已按 GB18030 编码读取 TXT 文件。"
                    )
                ]
            )
            return text, notes
        except UnicodeDecodeError:
            continue
    raise AgentError("TEXT_ENCODING_ERROR", "无法识别 TXT 编码，请另存为 UTF-8 后重试。")


def _parse_pdf(content: bytes, config: AgentConfig) -> tuple[str, int, list[WarningInfo]]:
    if b"%PDF-" not in content[:1024]:
        raise AgentError("INVALID_FILE", "文件内容与 PDF 格式不符。")
    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            raise AgentError("ENCRYPTED_PDF", "请先解除 PDF 密码保护后再上传。")
        pages = len(reader.pages)
        if pages > config.max_pdf_pages:
            raise AgentError("TOO_MANY_PAGES", f"PDF 超过 {config.max_pdf_pages} 页的处理上限。")
        parts = []
        blank = 0
        size = 0
        for page in reader.pages:
            text = page.extract_text() or ""
            size += len(text)
            if size > config.max_text_chars:
                raise AgentError("TEXT_TOO_LONG", "提取文本超过处理上限，请上传摘要或缩短文档。")
            blank += not bool(text.strip())
            parts.append(text)
        if not any(part.strip() for part in parts):
            raise AgentError(
                "NO_EXTRACTABLE_TEXT", "PDF 没有可提取文本，可能需要 OCR；也可直接粘贴正文。"
            )
        notes = []
        if blank:
            notes.append(
                WarningInfo(
                    code="PARTIAL_PDF_TEXT",
                    message=f"有 {blank} 页未提取到文本，分类仅使用其余页面。",
                )
            )
        return "\n\n".join(parts), pages, notes
    except AgentError:
        raise
    except Exception as exc:
        raise AgentError("DOCUMENT_PARSE_FAILED", "PDF 解析失败，请检查文件是否损坏。") from exc


def _parse_docx(content: bytes, config: AgentConfig) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            entries = archive.infolist()
            if (
                len(entries) > config.max_docx_entries
                or sum(item.file_size for item in entries) > config.max_docx_uncompressed_bytes
            ):
                raise AgentError("FILE_TOO_LARGE", "DOCX 解压后的内容超过处理上限。")
            if "word/document.xml" not in archive.namelist():
                raise AgentError("INVALID_FILE", "文件内容与 DOCX 格式不符。")
        document = Document(io.BytesIO(content))
        parts: list[str] = []
        size = 0
        # Preserve the order of paragraphs and tables in the document body.
        for block in document.iter_inner_content():
            if isinstance(block, Paragraph):
                text = block.text
            elif isinstance(block, Table):
                text = "\n".join("\t".join(cell.text for cell in row.cells) for row in block.rows)
            else:
                continue
            size += len(text)
            if size > config.max_text_chars:
                raise AgentError("TEXT_TOO_LONG", "提取文本超过处理上限，请上传摘要或缩短文档。")
            parts.append(text)
        return "\n".join(parts)
    except AgentError:
        raise
    except Exception as exc:
        raise AgentError("DOCUMENT_PARSE_FAILED", "DOCX 解析失败，请检查文件是否损坏。") from exc


def parse_document(content: bytes, filename: str, config: AgentConfig) -> ParsedDocument:
    if not isinstance(content, bytes) or not content:
        raise AgentError("EMPTY_FILE", "上传文件为空。")
    if len(content) > config.max_file_bytes:
        raise AgentError("FILE_TOO_LARGE", "上传文件超过大小限制。")
    # Filename is display metadata, never used as a filesystem destination.
    filename = PurePosixPath(filename.replace("\\", "/")).name
    suffix = PurePosixPath(filename).suffix.lower()
    notes: list[WarningInfo] = []
    pages = None
    if suffix == ".txt":
        text, notes = _decode_text(content)
    elif suffix == ".pdf":
        text, pages, notes = _parse_pdf(content, config)
    elif suffix == ".docx":
        text = _parse_docx(content, config)
    else:
        raise AgentError("UNSUPPORTED_FILE_TYPE", "请上传 TXT、PDF 或 DOCX 文件。")
    if len(text) > config.max_text_chars:
        raise AgentError("TEXT_TOO_LONG", "提取文本超过处理上限，请上传摘要或缩短文档。")
    return ParsedDocument(
        text=text, source_type=suffix[1:], filename=filename, pages=pages, warnings=notes
    )
