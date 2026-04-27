import re

from .state import IngestionState


def clean_node(state: IngestionState):
    text = state["raw_text"]
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"-{4,}", "--", text)
    text = re.sub(r"_{4,}", "__", text)
    text = re.sub(r"^## (?!\d+\.\d+\b)(.*)", r"### \1", text, flags=re.MULTILINE)
    text = re.sub(r"(?m)^### (.+)\n\s*### (\d+\.(?:\d+)?\s+.*)", r"# \1 - \2", text)
    text = re.sub(r"^###\s*(\d+.*?)$", r"# \1", text, flags=re.MULTILINE)

    text = re.sub(r"(?m)^### (\d+\.\d+)\s+(.*)", r"## \1 \2", text)
    text = re.sub(r"\n\n(?=- )", "\n", text)
    text = re.sub(r"\n\n\s+(?=- )", " ", text)
    text = re.sub(r"(?m)^(\s*-\s+)\S+\s+", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return {"cleaned_text": text}
