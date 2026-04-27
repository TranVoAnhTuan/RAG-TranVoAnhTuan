import json

path = "/home/jacktran/RAG/experiment/rag_agentic_system/evaluation/testset_updated.json"

with open(path, encoding="utf-8") as f:
    data = json.load(f)

for i, item in enumerate(data):
    if i <= 56:
        item["topic"] = "Insurance"
    else:
        item["topic"] = "QNU"

with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"Updated {len(data)} items in {path}")
