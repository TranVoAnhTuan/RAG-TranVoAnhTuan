from loguru import logger
from zenml import step, get_step_context
from typing_extensions import Annotated

from llm_engineering.application.crawlers.extract_pdf import PDFTextExtractor


@step
def extract_pdf_step(
    pdf_path: list[str],
) -> Annotated[list[str], "pdf_paths"]:
    logger.info(f"Starting PDF extraction step: {pdf_path}")

    extractor = PDFTextExtractor()

    for path in pdf_path:
        extractor.extract(path)

    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="pdf_paths",
        metadata=_add_to_metadata(pdf_path),
    )

    return pdf_path


def _add_to_metadata(pdf_paths: list[str]) -> dict:
    return {
        "source": [
            {
                "type": "pdf",
                "path": path,
            }
            for path in pdf_paths
        ],
        "operation": {
            "action": "extract_and_store",
        },
    }
