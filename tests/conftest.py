import io

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from agent import create_demo_agent


@pytest.fixture
def demo_agent():
    return create_demo_agent()


@pytest.fixture
def cardiac_text():
    return "分析高血压患者心血管风险与冠心病的关系，并比较不同干预措施的长期效果。"


def make_pdf(text: str | None = None, *, encrypted: bool = False, blank_pages: int = 0):
    writer = PdfWriter()
    page = writer.add_blank_page(width=595, height=842)
    if text:
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
        )
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 40 750 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    for _ in range(blank_pages):
        writer.add_blank_page(width=595, height=842)
    if encrypted:
        writer.encrypt("test-password")
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()
