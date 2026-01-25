from typing_extensions import Annotated
from zenml import get_step_context, step
from loguru import logger

from llm_engineering.domain.documents import PDFTextDocument


@step
def query_data_warehouse() -> Annotated[list, "raw_documents"]:
    logger.info("Querying all PDF text documents from MongoDB")

    documents = PDFTextDocument.bulk_find()

    if not documents:
        logger.warning("No PDF documents found in database")

    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="raw_documents",
        metadata=_get_metadata(documents),
    )

    return documents


def _get_metadata(documents: list) -> dict:
    return {
        "num_documents": len(documents),
        "sources": list({doc.path for doc in documents}),
    }
