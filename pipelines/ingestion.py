from zenml import pipeline
from pipelines.steps import extract_pdf_step, cleaning_step, chunking_step, embedding_and_load_step

@pipeline
def process_pdf_pipeline(file_path: str):
    raw_text = extract_pdf_step(file_path)
    cleaned_text = cleaning_step(raw_text)
    chunks = chunking_step(cleaned_text)
    result = embedding_and_load_step(chunks)
    return result