from loguru import logger
from zenml import step
from typing_extensions import Annotated
from zenml import get_step_context

from llm_engineering.application.crawlers.extract_pdf import PDFTextExtractor


@step
def extract_pdf_step(
    pdf_path: str,
) -> Annotated[str, "pdf_path"]:
    logger.info(f"Starting PDF extraction step: {pdf_path}")

    extractor = PDFTextExtractor()
    extractor.extract(pdf_path)

    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="pdf_path",
        metadata=_add_to_metadata(pdf_path),
    )

    return pdf_path

def _add_to_metadata(pdf_path: str) -> dict:
    return {
        "source": {
            "type": "pdf",
            "path": pdf_path,
        },
        "operation": {
            "action": "extract_and_store",
        },
    }