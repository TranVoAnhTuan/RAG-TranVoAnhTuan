import asyncio
import os
import re
import gc
import pandas as pd

from typing import TypedDict, List

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorOptions


# =========================
# STATE
# =========================
class IngestionState(TypedDict):
    file_path: str
    raw_text: str
    cleaned_text: str
    chunks: List[dict]
    tables: List[dict]
    status: str


# =========================
# DOC LING
# =========================
def _process_with_docling(file_path: str) -> dict:
    print("Loading Docling...")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True

    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=6,
        device="cpu"
    )

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    result = converter.convert(file_path)
    doc = result.document

    tables_data = []
    if hasattr(doc, 'tables'):
        for i, table in enumerate(doc.tables, 1):
            try:
                df = table.export_to_dataframe()

                df = df.replace(r'^[-|\s]*$', pd.NA, regex=True)
                df = df.dropna(how='all', axis=0)
                df = df.fillna("")
                if df.empty:
                    continue

                tables_data.append({
                    "table_number": i,
                    "data": df.to_dict(orient="records")
                })
            except Exception as e:
                print(f"Table error {i}: {e}")

    markdown_text = doc.export_to_markdown()

    print("Cleaning RAM...")
    del converter, result, doc
    gc.collect()

    return {
        "markdown": markdown_text,
        "tables": tables_data
    }


# =========================
# CLEAN
# =========================
def clean_text(text: str) -> str:
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    text = re.sub(r'(?m)^\s*-{3,}\s*$', '', text)
    text = re.sub(r'^## (?!\d+\.\d+\b)(.*)', r'### \1', text, flags=re.MULTILINE)
    text = re.sub(r'(?m)^### (.+)\n### (.+)', r'# \1 - \2', text)
    text = re.sub(r'(?m)^### (\d+\.\d+)\s+(.*)', r'## \1 \2', text)

    text = re.sub(r'\n\n(?=- )', '\n', text)
    text = re.sub(r'\n\n\s+(?=- )', ' ', text)
    text = re.sub(r'(?m)^(\s*-\s+)\S+\s+', r'\1', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


# =========================
# MAIN TEST
# =========================
async def main():
    file_path = "/home/jacktran/RAG/experiment/rag_agentic_system/Aust - Underwriting Guidelines (Prime) 1.pdf"  # <-- đổi path file của bạn

    if not os.path.exists(file_path):
        print("File not found")
        return

    result = await asyncio.to_thread(_process_with_docling, file_path)

    raw_text = result["markdown"]
    cleaned_text = clean_text(raw_text)

    # ghi file markdown
    with open("/home/jacktran/RAG/experiment/rag_agentic_system/test_markdown.md", "w", encoding="utf-8") as f:
        f.write(raw_text)

    with open("output_cleaned.md", "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    print("Done!")
    print(f"Tables extracted: {len(result['tables'])}")


if __name__ == "__main__":
    asyncio.run(main())