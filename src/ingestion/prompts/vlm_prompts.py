"""
FinSight AI — VLM Prompts
Prompt templates cho trích xuất tài liệu tài chính đa ngôn ngữ.
"""

from __future__ import annotations

from src.ingestion.schemas import OCRBlock


def format_ocr_blocks(ocr_blocks: list[OCRBlock], max_blocks: int = 300) -> str:
    """Format OCR blocks thành context ngắn gọn cho VLM."""
    if not ocr_blocks:
        return "(không có OCR blocks)"

    lines: list[str] = []
    for idx, block in enumerate(ocr_blocks[:max_blocks], start=1):
        confidence = f"{block.confidence:.2f}"
        lines.append(f"{idx}. [{confidence}] {block.text}")
    return "\n".join(lines)


def _profile_rules(prompt_profile: str) -> str:
    """Supplemental rules based on layout route."""
    profile = prompt_profile.upper().strip()
    if profile == "TEXT_PRIORITY":
        return """
PROFILE TEXT_PRIORITY:
- The document is primarily text. Focus on reading the main paragraphs and logical flow.
- If any small tables or key-value structures appear, you MUST still render them as Markdown tables. Do NOT discard them.
- Preserve the linear paragraph and heading order exactly as in the image.
""".strip()
    if profile == "TABLE_PRIORITY":
        return """
PROFILE TABLE_PRIORITY:
- The document is primarily tabular or numeric. Construct Markdown pipe tables representing the visible structure with highest priority.
- You MUST also extract any surrounding context (Company Name, Tax Code, Invoice Number, Date, Total Amount in words, etc.) before and after the table.
- DO NOT translate headers, normalize labels, or auto-correct typos in the data.
- If unsure about a row/column cell, preserve the raw token and mark it as `[uncertain: ...]`.
""".strip()
    if profile == "ANTI_HALLUCINATION_MIXED":
        return """
PROFILE ANTI_HALLUCINATION_MIXED:
- The image is noisy or blurry. ANTI-HALLUCINATION PROTOCOL ACTIVATED.
- You must strictly copy EXACTLY what is visible. If you cannot read a word, output `[unreadable]`.
- DO NOT infer, guess, or calculate any numbers. Keep layout mixed (tables and text).
""".strip()
    return """
PROFILE MIXED_LAYOUT:
- The document contains both text and tables. Maintain order: primary titles/info first, then tables.
- Do not mix regular text into table rows unless visually represented that way.
- If the table is ambiguous, output raw text in Markdown and mark as uncertain.
""".strip()


def build_financial_extraction_prompt(
    ocr_blocks: list[OCRBlock] | None = None,
    *,
    prompt_profile: str = "MIXED_LAYOUT",
) -> str:
    """Markdown-first prompt for VLM, applicable to all document types."""
    ocr_context = format_ocr_blocks(ocr_blocks or [])
    profile_rules = _profile_rules(prompt_profile)
    return f"""
You are a highly capable local, multilingual document OCR system.
Your ONLY task is to read the provided image and output strict, clean, structured Markdown for RAG ingestion.

!!! CRITICAL INSTRUCTIONS !!!
- Output ONLY pure Markdown. NO JSON formats.
- DO NOT explain, summarize, or add conversational filler.
- Preserve the ORIGINAL LANGUAGE exactly as seen in the image. Do NOT translate or infer intent.
- MUST output the exact spelling from the image, including all accents, diacritics, and special characters. DO NOT output unaccented or simplified text if accents are visible in the image.
- Do NOT hallucinate data, perform math calculations, or silently correct uncertain numbers.
- If a word/number is illegible or uncertain, write `[uncertain: <best_guess>]` at that position.

OUTPUT OBJECTIVES:
1. Preserve the original document structure as perfectly as possible.
2. Prioritize correct headings, paragraphs, bullet lists, tables, and key-value pairs.
3. If a table exists, construct a well-formatted Markdown pipe table.
4. For forms/receipts/invoices, group fields logically without forcing a fixed schema.
5. For plain text, preserve paragraphs and headings; do NOT invent fake tables.

MARKDOWN RULES:
- Use `#`, `##`, `###` for titles and sections.
- Use `- Label: value` for standalone fields.
- Use Markdown tables for clearly gridded data.
- Retain exact currencies, IDs, and date formats.
- Do not normalize formats unless clearly indicated in the image.

{profile_rules}

SUPPLEMENTAL OCR TEXT BLOCKS (from OCR Engine):
WARNING: The text blocks below lack language-specific accents and diacritics, BUT their base letters are highly accurate.
CRITICAL ANTI-HALLUCINATION RULES:
1. You MUST use the provided OCR text as your strict structural and foundational base.
2. Your ONLY permitted modification is restoring missing accents, diacritics, or language-specific characters (e.g., if OCR text is "HELLO", you may output "HÊLLÖ" if visible in the image).
3. NEVER replace a base word from the OCR text with a completely different sequence of letters.
4. NEVER hallucinate, invent, or guess names, addresses, or numerical values. If an area is blurry, rely entirely on the exact letters provided in the OCR block.

{ocr_context}
""".strip()
