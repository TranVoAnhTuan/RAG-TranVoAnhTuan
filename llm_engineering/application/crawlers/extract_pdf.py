from loguru import logger
from pdfminer.high_level import extract_text

from llm_engineering.domain.documents import PDFTextDocument
from .base import BasePDFExtractor


class PDFTextExtractor(BasePDFExtractor):
    model = PDFTextDocument

    def extract(self, pdf_path: str, **kwargs) -> None:
        old = self.model.find(path=pdf_path)
        if old is not None:
            logger.info(f"PDF already exists in database: {pdf_path}")
            return

        logger.info(f"Extracting PDF text: {pdf_path}")

        text = extract_text(pdf_path)

        instance = self.model(
            path=pdf_path,
            text=text,
        )
        instance.save()

        logger.info(f"Successfully extracted and saved PDF: {pdf_path}")
