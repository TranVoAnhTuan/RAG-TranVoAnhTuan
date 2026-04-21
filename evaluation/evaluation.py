import os
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
# from langchain_ollama import ChatOllama
from langsmith import Client
from langsmith.evaluation import evaluate
import json
# from app.core.config import settings

# Set API keys
os.environ["LANGCHAIN_API_KEY"] = 
# os.environ["GOOGLE_API_KEY"] = "..." 


class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    score: Annotated[float, ..., "A score from 0 to 1 based on factual accuracy."]

correctness_instructions = """You are a teacher grading a quiz. You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. 
(2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the ground truth answer.
Correctness:
Give a score between 0.0 and 1.0. 
1.0 is perfectly accurate.
0.0 is completely wrong.
Intermediate values (0.1 - 0.9) can be used for partially correct answers.
Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.
Output your response as a JSON object with the keys 'explanation' and 'score'."""

grader_llm = ChatOpenAI(
    model="gpt-oss:120b-cloud",
    temperature=0,
    base_url="http://host.docker.internal:11434/v1",
    api_key="ollama",
    model_kwargs={"response_format": {"type": "json_object"}}
).with_structured_output(CorrectnessGrade)

def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> float:
    answers = f"""\
QUESTION: {inputs['question']}
GROUND TRUTH ANSWER: {reference_outputs['answer']}
STUDENT ANSWER: {outputs['answer']}"""
    
    grade = grader_llm.invoke([
        {"role": "system", "content": correctness_instructions},
        {"role": "user", "content": answers}
    ])
    return grade["score"]


class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    score: Annotated[float, ..., "A score from 0 to 1 based on relevance."]

relevance_instructions = """You are a teacher grading a quiz. You will be given a QUESTION and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION
Relevance:
Give a score between 0.0 and 1.0.
1.0 is perfectly relevant.
0.0 is completely irrelevant.
Intermediate values (0.1 - 0.9) can be used for partially relevant answers.
Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.
Output your response as a JSON object with the keys 'explanation' and 'score'."""

relevance_llm = ChatOpenAI(
    model="gpt-oss:120b-cloud",
    temperature=0,
    base_url="http://host.docker.internal:11434/v1",
    api_key="ollama",
    model_kwargs={"response_format": {"type": "json_object"}}
).with_structured_output(RelevanceGrade)

def relevance(inputs: dict, outputs: dict) -> float:
    answer = f"QUESTION: {inputs['question']}\nSTUDENT ANSWER: {outputs['answer']}"
    grade = relevance_llm.invoke([
        {"role": "system", "content": relevance_instructions},
        {"role": "user", "content": answer}
    ])
    return grade["score"]


# ============ MAIN EVALUATION ============

# Load JSON data
with open("evaluation/testset_with_answers.json", "r", encoding="utf-8") as f:
    data = json.load(f)

client = Client()
dataset_name = "rag-evaluation-dataset"

try:
    client.delete_dataset(dataset_name=dataset_name)
except:
    pass

dataset = client.create_dataset(
    dataset_name, 
    description="RAG evaluation dataset"
)

for item in data:
    client.create_example(
        inputs={"question": item["question"]},
        outputs={"answer": item["ground_truth"]},  # Ground truth ở đây
        dataset_id=dataset.id
    )

answer_map = {item["question"]: item["answer"] for item in data}

def predict(inputs: dict) -> dict:
    question = inputs["question"]
    return {"answer": answer_map.get(question, "")}

results = evaluate(
    predict,
    data=dataset_name,
    evaluators=[correctness, relevance],
    experiment_prefix="rag-eval",
)

print("\n" + "="*50)
print("EVALUATION SUMMARY")
print("="*50)
print(f"Total examples: {len(data)}")
print(f"\nResults available at: https://smith.langchain.com")