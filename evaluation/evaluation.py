import os
from typing_extensions import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import Client
from langsmith.evaluation import evaluate
import json
from app.core.config import settings

# Set API keys
os.environ["LANGCHAIN_API_KEY"] = 
os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY


# ============ ĐỊNH NGHĨA EVALUATORS ============

class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]

correctness_instructions = """You are a teacher grading a quiz. You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. 
(2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the ground truth answer.
Correctness:
A correctness value of True means that the student's answer meets all of the criteria.
A correctness value of False means that the student's answer does not meet all of the criteria.
Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct."""

# Dùng Gemini thay GPT-4
grader_llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0
).with_structured_output(CorrectnessGrade)

def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    answers = f"""\
QUESTION: {inputs['question']}
GROUND TRUTH ANSWER: {reference_outputs['answer']}
STUDENT ANSWER: {outputs['answer']}"""
    
    grade = grader_llm.invoke([
        {"role": "system", "content": correctness_instructions},
        {"role": "user", "content": answers}
    ])
    return grade["correct"]


class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[bool, ..., "Provide the score on whether the answer addresses the question"]

relevance_instructions = """You are a teacher grading a quiz. You will be given a QUESTION and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION
Relevance:
A relevance value of True means that the student's answer meets all of the criteria.
A relevance value of False means that the student's answer does not meet all of the criteria.
Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct."""

relevance_llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0
).with_structured_output(RelevanceGrade)

def relevance(inputs: dict, outputs: dict) -> bool:
    answer = f"QUESTION: {inputs['question']}\nSTUDENT ANSWER: {outputs['answer']}"
    grade = relevance_llm.invoke([
        {"role": "system", "content": relevance_instructions},
        {"role": "user", "content": answer}
    ])
    return grade["relevant"]


# ============ MAIN EVALUATION ============

# Load JSON data
with open("/home/jacktran/RAG/experiment/rag_agentic_system/evaluation/testset_with_answers.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Tạo dataset trong LangSmith
client = Client()
dataset_name = "rag-evaluation-dataset"

# Xóa dataset cũ nếu có
try:
    client.delete_dataset(dataset_name=dataset_name)
except:
    pass

# Tạo dataset mới
dataset = client.create_dataset(
    dataset_name, 
    description="RAG evaluation dataset"
)

# Upload examples
for item in data:
    client.create_example(
        inputs={"question": item["question"]},
        outputs={"answer": item["ground_truth"]},  # Ground truth ở đây
        dataset_id=dataset.id
    )

# Tạo dictionary để map question -> answer
answer_map = {item["question"]: item["answer"] for item in data}

# Hàm trả về answer đã có sẵn
def predict(inputs: dict) -> dict:
    question = inputs["question"]
    return {"answer": answer_map.get(question, "")}

# Chạy evaluation
results = evaluate(
    predict,
    data=dataset_name,
    evaluators=[correctness, relevance],
    experiment_prefix="rag-eval",
)

# In kết quả tổng hợp
print("\n" + "="*50)
print("EVALUATION SUMMARY")
print("="*50)
print(f"Total examples: {len(data)}")
print(f"\nResults available at: https://smith.langchain.com")