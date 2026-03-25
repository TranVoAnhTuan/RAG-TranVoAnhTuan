import os
import gc
import asyncio
import pandas as pd
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorOptions
from .state import IngestionState

def _process_with_docling(file_path: str) -> dict:
    print("Loading Docling...")
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.accelerator_options = AcceleratorOptions(num_threads=6, device="cpu")

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
                df = df.replace(r'^[-|\s]*$', pd.NA, regex=True).dropna(how='all', axis=0).fillna("")
                if not df.empty:
                    tables_data.append({"table_number": i, "data": df.to_dict(orient="records")})
            except Exception as e:
                print(f"Table error {i}: {e}")

    markdown_text = doc.export_to_markdown()

    # Dọn dẹp RAM
    print("Cleaning Docling RAM...")
    del converter, result, doc
    gc.collect()

    return {"markdown": markdown_text, "tables": tables_data}

async def extract_node(state: IngestionState):
    file_path = state["file_path"]
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No file found at: {file_path}")
    
    docling_result = await asyncio.to_thread(_process_with_docling, file_path)
    
    return {
        "raw_text": docling_result["markdown"],
        "tables": docling_result["tables"] 
    }