from abc import ABC, abstractmethod
from llm_engineering.domain.documents import NoSQLBaseDocument


class BasePDFExtractor(ABC):
    model: type[NoSQLBaseDocument]

    @abstractmethod
    def extract(self, pdf_path: str, **kwargs) -> None:
        pass