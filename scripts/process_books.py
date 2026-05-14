import json
import re
from pathlib import Path

from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "data" / "textbooks"
OUTPUT_DIR = BASE_DIR / "processed"
OUTPUT_FILE = OUTPUT_DIR / "chunks.json"

MIN_CHARS = 500
MAX_CHARS = 1500
OVERLAP_CHARS = 120

# Paragraph split for oversized blocks inside build_semantic_chunks path
LONG_PARAGRAPH_MAX = 750

# Safety: never strip more than this many leading paragraphs as "front matter"
MAX_LEADING_STRIPS = 220


def detect_book_language(filename: str) -> str:
    name = filename.lower()

    if "_tr" in name or name.endswith("tr.pdf") or "turk" in name or "türk" in name:
        return "tr"

    if "_en" in name or name.endswith("en.pdf") or "english" in name:
        return "en"

    return "en"


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.replace("\u00ad", "")
    text = text.replace("\u2010", "-")
    text = text.replace("\u2011", "-")
    text = text.replace("\u2013", "-")
    text = text.replace("\u2014", "-")
    text = text.replace("\ufb01", "fi")
    text = text.replace("\ufb02", "fl")
    return text.strip()


def join_hyphenated_line_breaks(text: str) -> str:
    """Join words split across a line break with a hyphen (triamci-\\nnolone)."""
    return re.sub(r"(?<=[a-zA-Z])-\s*\n\s*(?=[a-zA-Z])", "", text)


def fix_common_pdf_space_splits(text: str) -> str:
    """Fix frequent column-merge artifacts (conservative, word-boundary only)."""
    pairs = [
        (r"\bt\s+he\b", "the"),
        (r"\bT\s+he\b", "The"),
        (r"\ba\s+nd\b", "and"),
        (r"\bi\s+n\b", "in"),
        (r"\bo\s+f\b", "of"),
        (r"\bi\s+t\b", "it"),
        (r"\bt\s+o\b", "to"),
        (r"\bf\s+or\b", "for"),
        (r"\ba\s+t\b", "at"),
        (r"\bi\s+s\b", "is"),
        (r"\bo\s+n\b", "on"),
        (r"\ba\s+n\b", "an"),
        (r"\bo\s+r\b", "or"),
        (r"\bw\s+ith\b", "with"),
        (r"\bW\s+ith\b", "With"),
        (r"\ba\s+s\b", "as"),
        (r"\bb\s+y\b", "by"),
        (r"\bd\s+ermatology\b", "dermatology"),
        (r"\bD\s+ermatology\b", "Dermatology"),
    ]
    for pat, repl in pairs:
        text = re.sub(pat, repl, text)
    return text


def strip_running_header_lines(paragraph: str) -> str:
    """Remove short running heads / page furniture lines."""
    lines_out = []
    for line in paragraph.split("\n"):
        s = line.strip()
        if not s:
            lines_out.append(line)
            continue
        if len(s) <= 120 and re.match(r"(?i)^part\s+\d+:\s*$", s):
            continue
        if len(s) <= 80 and re.match(r"(?i)^contents\s+[ivxlcdm]+\s*$", s):
            continue
        if len(s) <= 80 and re.match(r"(?i)^[ivxlcdm]+\s+contents\s*$", s):
            continue
        if len(s) <= 120 and re.match(r"(?i)^contents[ivxlcdm\d]*\s*$", s):
            continue
        if len(s) <= 160 and re.match(
            r"(?i)^\d+\.\d+\s+chapter\s+\d+:", s
        ):
            continue
        if re.match(r"(?i)^index\s*$", s) and len(s) < 20:
            continue
        lines_out.append(line)
    return "\n".join(lines_out).strip()


def _consume_caption_block(lines: list[str], start: int, starters: tuple[re.Pattern, ...]) -> int:
    """Advance index past a caption block starting at start (through following non-empty lines)."""
    i = start + 1
    while i < len(lines):
        nxt = lines[i].strip()
        if nxt == "":
            return i + 1
        if any(p.match(nxt) for p in starters):
            return i
        i += 1
    return i


def remove_figure_caption_lines(paragraph: str) -> str:
    """Drop Figure / Table caption blocks (multi-line, Courtesy-of, subfigure labels)."""
    fig_starters = (
        re.compile(r"(?i)^Figure\s+\d+\.\d+"),
        re.compile(r"(?i)^Figure\s+\d+\b"),
    )
    table_starters = (
        re.compile(r"(?i)^Tablo\s+\d+"),
        re.compile(r"(?i)^Table\s+\d+"),
    )
    lines = paragraph.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        if any(p.match(stripped) for p in fig_starters):
            i = _consume_caption_block(lines, i, fig_starters)
            continue
        if any(p.match(stripped) for p in table_starters):
            i = _consume_caption_block(lines, i, table_starters)
            continue
        if re.match(r"(?i)^\([a-z]\)\s*$", stripped) and out and (
            "Figure" in "\n".join(out[-4:]) or "Tablo" in "\n".join(out[-4:])
        ):
            i += 1
            continue
        out.append(raw)
        i += 1
    return "\n".join(out).strip()


def is_reference_paragraph(p: str) -> bool:
    head = p.strip()[:220]
    first_line = head.split("\n")[0].strip()
    if re.match(r"(?i)^key references\b", head):
        return True
    if re.match(r"(?i)^references\s*$", first_line):
        return True
    if re.match(r"(?i)^kaynaklar\b", first_line):
        return True
    if re.match(r"(?i)^bibliography\b", first_line):
        return True
    if re.match(r"(?i)^bibliografya\b", first_line):
        return True
    if re.match(r"(?i)^kaynakça\b", first_line):
        return True
    lines = [ln for ln in p.split("\n") if ln.strip()]
    if len(lines) < 4:
        return False

    def line_looks_like_citation(ln: str) -> bool:
        s = ln.strip()
        if re.match(r"^\s*\d+\s+[A-ZÀ-ŸÇĞİÖŞÜ]", s):
            return True
        if re.match(r"^\s*\d+\s+[A-Z][a-z]+,\s+[A-Z]", s):
            return True
        if re.match(r"^\s*\d+\.\s+", s):
            tail = s[5:120] if len(s) > 5 else s
            return bool(
                re.search(r"\d{4}", s)
                or re.search(r"\bet al\b", s, re.I)
                or re.search(r"\bve\s+ark\.?", s, re.I)
                or re.search(r"\bdoi:\s*", s, re.I)
                or ";" in tail
                or re.search(r"\b(pp?\.|vol\.|issue)\b", s, re.I)
            )
        return False

    cite_like = sum(1 for ln in lines if line_looks_like_citation(ln))
    ref_signals = bool(
        re.search(r"\bet al\b", p, re.I)
        or re.search(r"\bve\s+ark\.?\b", p, re.I)
        or re.search(r"\beds?\.?\s+", p, re.I)
        or re.search(r"\bdoi:\s*", p, re.I)
    )
    if cite_like >= max(4, int(len(lines) * 0.45)) and ref_signals:
        return True
    return False


def is_wiley_chapter_banner_paragraph(p: str) -> bool:
    pl = p
    if len(pl) > 1400:
        return False
    if not re.search(r"(?is)Rook'?s\s+Textbook\s+of\s+Dermatology", pl):
        return False
    if not re.search(r"(?is)John\s+Wiley", pl):
        return False
    return True


def is_toc_line_topic_then_page(ln: str) -> bool:
    """Atlas-style TOC: 'Psoriatic Erythroderma 67' (title + page, no sentence punctuation)."""
    s = ln.strip()
    if len(s) < 12 or len(s) > 110:
        return False
    if re.search(r"[.!?:;]", s):
        return False
    return bool(
        re.match(
            r"^[A-Za-zÀ-ÿĞÜŞİÖÇğüşıöç0-9][A-Za-zÀ-ÿĞÜŞİÖÇğüşıöç0-9\s\-,/'()+]+\s+\d{1,4}$",
            s,
        )
    )


def is_fitzpatrick_style_toc(p: str) -> bool:
    """Multi-line contents where most lines are section/topic + trailing page number."""
    if re.search(r"(?im)^CONTENTS", p):
        lines = [ln.strip() for ln in p.split("\n") if ln.strip()]
        if len(lines) >= 8:
            topic_page = sum(1 for ln in lines if is_toc_line_topic_then_page(ln))
            section_hits = sum(1 for ln in lines if re.match(r"(?i)^section\s+\d+", ln))
            part_hits = sum(1 for ln in lines if re.match(r"(?i)^part\s+[ivx]+\s*$", ln))
            if topic_page + section_hits + part_hits >= max(7, int(len(lines) * 0.38)):
                return True
            if topic_page >= 5 and section_hits >= 1:
                return True
    lines = [ln.strip() for ln in p.split("\n") if ln.strip()]
    if len(lines) < 10:
        return False
    topic_page = sum(1 for ln in lines if is_toc_line_topic_then_page(ln))
    section_hits = sum(1 for ln in lines if re.match(r"(?i)^section\s+\d+", ln))
    part_hits = sum(1 for ln in lines if re.match(r"(?i)^part\s+[ivx]+\s*$", ln))
    if topic_page + section_hits + part_hits >= max(8, int(len(lines) * 0.42)):
        return True
    return False


def is_toc_heavy_paragraph(p: str) -> bool:
    if is_fitzpatrick_style_toc(p):
        return True
    if re.search(r"(?im)^\s*Contents\s*$", p):
        return True
    lines = [ln for ln in p.split("\n") if ln.strip()]
    if len(lines) < 5:
        return False
    toclines = sum(
        1
        for ln in lines
        if re.match(r"^\s*\d+\s+.+,\s*\d+\.\d+\s*$", ln)
        or re.match(r"(?i)^volume\s+\d+", ln.strip())
        or re.match(r"(?i)^part\s+\d+", ln.strip())
    )
    return toclines >= max(5, int(len(lines) * 0.35))


def is_editorial_directory_paragraph(p: str) -> bool:
    head = p.strip()[:400].lower()
    if "list of associate editors" in head:
        return True
    if "list of contributors" in head:
        return True
    if head.startswith("associate editors"):
        return True
    if re.match(r"(?is)^contributors\s*\n", p.strip()):
        return True
    lines = [ln for ln in p.split("\n") if ln.strip()]
    if len(lines) < 12:
        return False
    degree_hits = sum(
        1
        for ln in lines
        if re.search(r"\b(MD|PhD|FRCP|MBA|BSc|MB\s+ChB|DPhil|DMSc)\b", ln)
    )
    chapter_tail = sum(1 for ln in lines if re.search(r"\bChapters?\s+[\d,\s-]+\s*$", ln))
    if degree_hits >= 10 and chapter_tail >= 3:
        return True
    return False


def is_boilerplate_or_marketing_paragraph(p: str) -> bool:
    pl = p.lower()
    if "scratch off the sticker" in pl and "pin" in pl:
        return True
    if re.search(r"(?is)online edition\s+included\s+with\s+book\s+purchase", pl):
        return True
    if len(p) < 260 and "isbn" in pl and ("wiley" in pl or "copyright" in pl):
        return True
    if len(p) < 400 and "all rights reserved" in pl and "wiley" in pl:
        return True
    return False


def is_low_value_publisher_front_paragraph(p: str) -> bool:
    """
    McGraw-Hill / Fitzpatrick-style front matter that is not useful for clinical RAG.
    Kept conservative: short blocks with strong publisher + legal cues.
    """
    if len(p) > 4200:
        return False
    u = p.upper()
    if "FITZPATRICK" in u and "SIXTH EDITION" in u:
        if "AVAILABLE TRANSLATIONS OF FITZPATRICK" in u:
            return True
        if "PROFESSOR AND CHAIRMAN EMERITUS" in u and "DEPARTMENT OF DERMATOLOGY" in u:
            if len(p) < 3200 and "COLOR ATLAS" in u:
                return True
    if "DEDICATED TO THOMAS B. FITZPATRICK" in u or "DEDICATED TO MARIAPAZ RAMOS" in u:
        if len(p) < 2800:
            return True
    if "MCGRAW-HILL" in u or "MCGRAW HILL" in u:
        if "COPYRIGHT" in u and len(p) < 4000:
            if any(
                x in u
                for x in (
                    "ISBN",
                    "MHID",
                    "TERMS OF USE",
                    "TRADEMARK SYMBOL",
                    "BULKSALES@MCGRAWHILL",
                )
            ):
                return True
        if "THE WORK IS PROVIDED" in u and "AS IS" in u and len(p) < 3500:
            return True
        if "NEITHER MCGRAW-HILL NOR ITS LICENSORS SHALL BE LIABLE" in u and len(p) < 2200:
            return True
        if p.strip().startswith("NOTICE") and "MEDICINE IS AN EVER-CHANGING SCIENCE" in u:
            return True
    if "EXCEPT AS PERMITTED UNDER THE COPYRIGHT" in u and "MCGRAW-HILL" in u:
        if len(p) < 3600 and "DECOMPILE" in u.upper():
            return True
    return False


def is_prose_start_paragraph(p: str) -> bool:
    """Heuristic: first substantial body paragraph (stops front-matter stripping)."""
    if len(p) < 380:
        return False
    if (
        is_toc_heavy_paragraph(p)
        or is_reference_paragraph(p)
        or is_editorial_directory_paragraph(p)
        or is_low_value_publisher_front_paragraph(p)
    ):
        return False
    if p.count(". ") >= 2 or re.search(r"[.!?]\s+[A-ZÀ-ŸÇĞİÖŞÜ]", p):
        return True
    return False


def strip_trailing_reference_suffix(text: str) -> str:
    """
    Remove a reference appendix glued after clinical prose in the same block.

    Requires a newline before the heading so in-line phrases are not matched.
    Do not run on whole-book strings: apply per paragraph and per final chunk only.
    """
    t = re.sub(
        r"(?is)\n\s*(?:Kaynaklar|Kaynakça|Key\s+references|References)\b\s*\n[\s\S]*\Z",
        "",
        text,
    )
    t = re.sub(
        r"(?is)^\s*(?:Kaynaklar|Kaynakça|Key\s+references|References)\b\s*\n[\s\S]*\Z",
        "",
        t,
    )
    return t.strip()


def strip_leading_front_matter(paragraphs: list[str]) -> list[str]:
    out = list(paragraphs)
    stripped = 0
    while (
        out
        and stripped < MAX_LEADING_STRIPS
        and not is_prose_start_paragraph(out[0])
    ):
        first = out[0]
        if (
            is_toc_heavy_paragraph(first)
            or is_editorial_directory_paragraph(first)
            or is_boilerplate_or_marketing_paragraph(first)
            or is_wiley_chapter_banner_paragraph(first)
            or is_reference_paragraph(first)
            or is_low_value_publisher_front_paragraph(first)
            or (
                len(first.strip()) < 320
                and (
                    re.search(r"(?i)\b(isbn|catalogue record|john wiley|wiley-blackwell)\b", first)
                    or re.search(r"(?i)\ball rights reserved\b", first)
                )
            )
        ):
            out.pop(0)
            stripped += 1
            continue
        break
    return out


def filter_and_clean_paragraphs(paragraphs: list[str]) -> list[str]:
    cleaned: list[str] = []
    for p in paragraphs:
        if not p.strip():
            continue
        if is_wiley_chapter_banner_paragraph(p):
            continue
        if is_reference_paragraph(p):
            continue
        if is_low_value_publisher_front_paragraph(p):
            continue
        if is_toc_heavy_paragraph(p):
            continue
        if is_boilerplate_or_marketing_paragraph(p) and len(p) < 500:
            continue
        p = strip_running_header_lines(p)
        p = remove_figure_caption_lines(p)
        p = clean_text(p)
        p = strip_trailing_reference_suffix(p)
        if p:
            cleaned.append(p)
    cleaned = strip_leading_front_matter(cleaned)
    return cleaned


def is_heading(paragraph: str) -> bool:
    p = paragraph.strip()
    if len(p) < 3:
        return False

    first = p.split("\n")[0].strip()

    if re.match(r"(?i)^chapter\s+\d+\b", first):
        return len(first) <= 200 or len(p) <= 260

    if re.match(r"(?i)^part\s+\d+\b", first) and len(first) <= 120:
        return True

    if re.match(r"^\d+\.\d+\s+Chapter\s+\d+:", first):
        return len(p) <= 300

    if re.match(r"^\d+\s+[A-ZÇĞİÖŞÜ]", first) and len(first) <= 100:
        return True

    if "\n" not in p and len(p) <= 120:
        if re.match(r"^\d+(\.\d+)*\s+[A-ZÇĞİÖŞÜ]", p):
            return True
        if re.match(r"^[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ0-9 \-,:()/]{3,}$", p):
            return True
        if p.endswith(":") and len(p) < 80:
            return True

    if "\n" in p and len(p) <= 220:
        lines = [ln.strip() for ln in p.split("\n") if ln.strip()]
        if len(lines) == 2:
            a, b = lines[0], lines[1]
            if (
                len(a) < 90
                and len(b) < 130
                and not re.search(r"[.!?]", a)
                and re.match(r"^[A-ZÇĞİÖŞÜ]", a)
                and re.match(r"^[A-Z][a-z]", b)
            ):
                return True

    return False


def split_into_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text)
    cleaned = []

    for part in parts:
        p = clean_text(part)
        if p:
            cleaned.append(p)

    return cleaned


def split_long_paragraph(paragraph: str, max_chars: int = LONG_PARAGRAPH_MAX) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]

    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    pieces = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                pieces.append(current)
            current = sentence

    if current:
        pieces.append(current)

    return pieces


def build_semantic_chunks(paragraphs: list[str]) -> list[dict]:
    chunks = []
    current_chunk = ""
    current_section = "Unknown section"

    for raw_paragraph in paragraphs:
        if is_heading(raw_paragraph):
            if current_chunk.strip():
                chunks.append({
                    "section_title": current_section,
                    "text": current_chunk.strip(),
                })
                current_chunk = ""

            current_section = raw_paragraph.strip()
            continue

        sub_parts = split_long_paragraph(raw_paragraph)

        for paragraph in sub_parts:
            candidate = f"{current_chunk}\n\n{paragraph}".strip() if current_chunk else paragraph

            if len(candidate) <= MAX_CHARS:
                current_chunk = candidate
            else:
                if current_chunk.strip():
                    chunks.append({
                        "section_title": current_section,
                        "text": current_chunk.strip(),
                    })

                    overlap = current_chunk[-OVERLAP_CHARS:] if len(current_chunk) > OVERLAP_CHARS else current_chunk
                    current_chunk = f"{overlap}\n\n{paragraph}".strip()
                else:
                    chunks.append({
                        "section_title": current_section,
                        "text": paragraph.strip(),
                    })
                    current_chunk = ""

    if current_chunk.strip():
        chunks.append({
            "section_title": current_section,
            "text": current_chunk.strip(),
        })

    merged = []
    for chunk in chunks:
        if merged and len(chunk["text"]) < MIN_CHARS:
            merged[-1]["text"] = f'{merged[-1]["text"]}\n\n{chunk["text"]}'.strip()
        else:
            merged.append(chunk)

    return merged


def trim_duplicate_overlap(prev_text: str, curr_text: str) -> str:
    """If curr starts with a long suffix of prev, trim that prefix from curr."""
    if not prev_text or not curr_text:
        return curr_text
    max_len = min(len(prev_text), len(curr_text), 450)
    for n in range(max_len, 39, -1):
        if prev_text.endswith(curr_text[:n]):
            return curr_text[n:].lstrip()
    return curr_text


def deduplicate_chunk_boundaries(chunks: list[dict]) -> list[dict]:
    for i in range(1, len(chunks)):
        chunks[i]["text"] = trim_duplicate_overlap(chunks[i - 1]["text"], chunks[i]["text"])
    return [c for c in chunks if c["text"].strip()]


def extract_chapter_id(text: str) -> str | None:
    sample = text[:650]
    m = re.search(r"(?i)\bchapter\s+(\d+)\b", sample)
    if m:
        return m.group(1)
    m = re.search(r"(?i)\bsection\s+(\d+)\b", sample)
    if m:
        return f"S{m.group(1)}"
    m = re.search(r"(?:^|\n)\s*(\d+)\.(\d+)\s+", sample)
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    return None


def discard_chunk_if_noise(text: str, section_title: str) -> bool:
    t = text.strip()
    if len(t) < 72:
        return True
    if section_title == "Unknown section" and re.match(r"(?is)^contents\b", t):
        return True
    if section_title == "Unknown section" and re.match(r"(?is)^CONTENTS", t):
        return True
    if is_low_value_publisher_front_paragraph(t):
        return True
    if is_fitzpatrick_style_toc(t):
        return True
    lines = [ln for ln in t.split("\n") if ln.strip()]
    if len(lines) >= 8:
        tocish = sum(
            1
            for ln in lines
            if re.match(r"^\s*\d+\s+.+,\s*\d+\.\d+\s*$", ln.strip())
        )
        if tocish / len(lines) > 0.62:
            return True
        topic_page = sum(1 for ln in lines if is_toc_line_topic_then_page(ln))
        section_hits = sum(1 for ln in lines if re.match(r"(?i)^section\s+\d+", ln.strip()))
        if topic_page + section_hits >= max(7, int(len(lines) * 0.4)):
            return True
    if is_editorial_directory_paragraph(t):
        return True
    if is_wiley_chapter_banner_paragraph(t):
        return True
    if is_reference_paragraph(t):
        return True
    return False


def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []

    for i, page in enumerate(reader.pages, start=1):
        try:
            extracted = page.extract_text()
        except Exception:
            extracted = ""

        if extracted:
            pages.append(extracted)

        if i % 20 == 0:
            print(f"  Read {i} pages from {pdf_path.name}...")

    raw = "\n\n".join(pages)
    raw = join_hyphenated_line_breaks(raw)
    raw = fix_common_pdf_space_splits(raw)
    return clean_text(raw)


def main():
    print(f"Looking for PDFs in: {INPUT_DIR}")
    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF file(s).")

    if not pdf_files:
        print("No PDF files found.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    global_chunk_id = 1

    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")

        book_language = detect_book_language(pdf_file.name)
        text = extract_pdf_text(pdf_file)

        if not text:
            print(f"  No extractable text found in {pdf_file.name}")
            continue

        paragraphs = split_into_paragraphs(text)
        paragraphs = filter_and_clean_paragraphs(paragraphs)
        semantic_chunks = build_semantic_chunks(paragraphs)
        semantic_chunks = deduplicate_chunk_boundaries(semantic_chunks)

        for ch in semantic_chunks:
            ch["text"] = strip_trailing_reference_suffix(ch["text"])

        kept = []
        for ch in semantic_chunks:
            if discard_chunk_if_noise(ch["text"], ch["section_title"]):
                continue
            kept.append(ch)
        semantic_chunks = kept

        print(f"  Created {len(semantic_chunks)} semantic chunks")

        for local_id, chunk in enumerate(semantic_chunks, start=1):
            all_chunks.append({
                "chunk_id": global_chunk_id,
                "source": pdf_file.name,
                "source_path": str(pdf_file),
                "local_chunk_id": local_id,
                "language": book_language,
                "section_title": chunk["section_title"],
                "chapter_id": extract_chapter_id(chunk["text"]),
                "text": chunk["text"],
            })
            global_chunk_id += 1

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Total chunks created: {len(all_chunks)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
