from zenml import pipeline
from steps.etl import extract_pdf_step


@pipeline
def pdf_ingestion_pipeline(pdf_path: str):
    extract_pdf_step(pdf_path)

