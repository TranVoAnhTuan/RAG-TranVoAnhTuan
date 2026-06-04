import asyncio
import gc
import logging
import multiprocessing
import os
import re
from concurrent.futures import ProcessPoolExecutor

import torch
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from .state import IngestionState

logger = logging.getLogger(__name__)


def log_vram(step_name: str) -> None:
    """Helper function to log VRAM capacity"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        logger.info(f"📊 [{step_name}] VRAM Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")


def _process_with_docling(file_path: str) -> str:
    logger.info("Loading Docling...")
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.ocr_options = EasyOcrOptions(lang=["vi", "en"])
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=6, device="cuda" if torch.cuda.is_available() else "cpu"
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(backend=PyPdfiumDocumentBackend, pipeline_options=pipeline_options)
        }
    )
    result = converter.convert(file_path)
    markdown_text = result.document.export_to_markdown()

    logger.info("Cleaning Docling RAM...")
    del converter, result, pipeline_options
    gc.collect()
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    log_vram("4. AFTER CLEANUP")
    return markdown_text


def _extract_title(raw_text: str) -> str:
    """Extract the first 5 ``##`` headings from Docling raw markdown as a document title."""
    heading_lines = re.findall(r"^##\s+(.+)", raw_text, re.MULTILINE)
    selected = heading_lines[:5]
    if not selected:
        return ""
    return " | ".join(selected)


async def extract_node(state: IngestionState) -> dict:
    file_path = state["file_path"]
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No file found at: {file_path}")

    loop = asyncio.get_running_loop()
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as executor:
        markdown = await loop.run_in_executor(executor, _process_with_docling, file_path)

    return {"raw_text": markdown, "title": _extract_title(markdown)}
