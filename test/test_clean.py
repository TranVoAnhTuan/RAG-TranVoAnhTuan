import re
import sys
import os


def clean_text(text: str) -> str:
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    text = re.sub(r'-{4,}', '--', text)
    text = re.sub(r'^## (?!\d+\.\d+\b)(.*)', r'### \1', text, flags=re.MULTILINE)
    text = re.sub(r'(?m)^### (.+)\n\s*### (\d+\.(?:\d+)?\s+.*)', r'# \1 - \2',text)    
    text = re.sub(r'(?m)^### (\d+\.\d+)\s+(.*)', r'## \1 \2', text)

    text = re.sub(r'\n\n(?=- )', '\n', text)
    text = re.sub(r'\n\n\s+(?=- )', ' ', text)
    text = re.sub(r'(?m)^(\s*-\s+)\S+\s+', r'\1', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def main():
    if len(sys.argv) < 2:
        print("Usage: python clean_markdown.py input.md [output.md]")
        return

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "cleaned.md"

    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return

    # read file
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    # clean
    cleaned = clean_text(text)

    # write file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(cleaned)

    print(f"✅ Cleaned file saved to: {output_file}")


if __name__ == "__main__":
    main()