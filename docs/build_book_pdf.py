from __future__ import annotations

from pathlib import Path
from textwrap import wrap


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "formal_rl_length_generalization_book.md"
TARGET = ROOT / "formal_rl_length_generalization_book.pdf"


PAGE_W = 612
PAGE_H = 792
MARGIN = 54
FONT_SIZE = 10.5
LINE_H = 14
MAX_CHARS = 92


def escape_pdf(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def markdown_to_lines(text: str) -> list[dict[str, object]]:
    lines: list[dict[str, object]] = []
    for para in text.splitlines():
        if para.startswith("# "):
            lines.append({"text": para[2:], "size": 20, "before": 0, "after": 1})
        elif para.startswith("## "):
            lines.append({"text": para[3:], "size": 14, "before": 1, "after": 1})
        elif para.strip() == "":
            lines.append({"text": "", "size": FONT_SIZE, "before": 0, "after": 0})
        else:
            stripped = para.strip()
            if stripped.startswith("- "):
                wrapped = wrap(stripped, width=MAX_CHARS - 2) or [stripped]
                lines.append({"text": wrapped[0], "size": FONT_SIZE, "before": 0, "after": 0})
                for extra in wrapped[1:]:
                    lines.append({"text": "  " + extra, "size": FONT_SIZE, "before": 0, "after": 0})
            else:
                for line in wrap(stripped, width=MAX_CHARS) or [""]:
                    lines.append({"text": line, "size": FONT_SIZE, "before": 0, "after": 0})
    return lines


def paginate(lines: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    usable = int((PAGE_H - 2 * MARGIN) // LINE_H)
    pages: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    used = 0
    for item in lines:
        needed = 1 + int(item.get("before", 0)) + int(item.get("after", 0))
        if current and used + needed > usable:
            pages.append(current)
            current = []
            used = 0
        for _ in range(int(item.get("before", 0))):
            current.append({"text": "", "size": FONT_SIZE})
            used += 1
        current.append(item)
        used += 1
        for _ in range(int(item.get("after", 0))):
            current.append({"text": "", "size": FONT_SIZE})
            used += 1
    if current:
        pages.append(current)
    return pages


def build_pdf(pages: list[list[dict[str, object]]]) -> bytes:
    objects: list[str] = []

    def add_obj(body: str) -> int:
        objects.append(body)
        return len(objects)

    font_obj = add_obj("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_objs: list[int] = []

    for page_index, page in enumerate(pages, start=1):
        y = PAGE_H - MARGIN
        stream_parts = ["BT\n"]
        for item in page:
            text = str(item.get("text", ""))
            size = float(item.get("size", FONT_SIZE))
            if text:
                stream_parts.append(f"/F1 {size:g} Tf\n")
                stream_parts.append(f"1 0 0 1 {MARGIN} {y:.2f} Tm\n")
                stream_parts.append(f"({escape_pdf(text)}) Tj\n")
            y -= LINE_H
        stream_parts.append("/F1 9 Tf\n")
        stream_parts.append(f"1 0 0 1 {PAGE_W - MARGIN - 80} {MARGIN - 24} Tm\n")
        stream_parts.append(f"(Page {page_index} of {len(pages)}) Tj\n")
        stream_parts.append("ET\n")
        stream = "".join(stream_parts)
        content_obj = add_obj(
            f"<< /Length {len(stream.encode('utf-8'))} >>\nstream\n{stream}endstream"
        )
        page_obj = add_obj(
            "<< /Type /Page /Parent PARENT_REF "
            f"/MediaBox [0 0 {PAGE_W} {PAGE_H}] "
            f"/Resources << /Font << /F1 {font_obj} 0 R >> >> "
            f"/Contents {content_obj} 0 R >>"
        )
        page_objs.append(page_obj)

    pages_obj_num = len(objects) + 1
    objects = [body.replace("PARENT_REF", f"{pages_obj_num} 0 R") for body in objects]
    kids = " ".join(f"{obj} 0 R" for obj in page_objs)
    pages_obj = add_obj(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_objs)} >>")
    catalog_obj = add_obj(f"<< /Type /Catalog /Pages {pages_obj} 0 R >>")

    pdf = "%PDF-1.4\n"
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(pdf.encode("utf-8")))
        pdf += f"{index} 0 obj\n{body}\nendobj\n"
    xref_offset = len(pdf.encode("utf-8"))
    pdf += f"xref\n0 {len(objects) + 1}\n"
    pdf += "0000000000 65535 f \n"
    for offset in offsets[1:]:
        pdf += f"{offset:010d} 00000 n \n"
    pdf += (
        "trailer\n"
        f"<< /Size {len(objects) + 1} /Root {catalog_obj} 0 R >>\n"
        "startxref\n"
        f"{xref_offset}\n"
        "%%EOF\n"
    )
    return pdf.encode("utf-8")


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    pages = paginate(markdown_to_lines(source))
    TARGET.write_bytes(build_pdf(pages))
    print(f"Wrote {TARGET}")
    print(f"Pages: {len(pages)}")


if __name__ == "__main__":
    main()
