import re


def clean_text(text: str) -> str:
    text = re.sub(r"\n{2,}", ". ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" . ", ". ")
    return text.strip()
