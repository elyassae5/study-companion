import json
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# SETUP — same as phase4
# =============================================================================

loader = TextLoader("data/notes.md", encoding="utf-8")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

llm = ChatAnthropic(model="claude-haiku-4-5-20251001")

print("Setup complete\n")

# =============================================================================
# GRAPH — same as phase4
# =============================================================================

class StudyState(TypedDict):
    question: str
    retrieved_chunks: list
    answer: str

def retrieve(state: StudyState):
    question = state["question"]
    results = vectorstore.similarity_search(question, k=3)
    chunks_list = [doc.page_content for doc in results]
    return {"retrieved_chunks": chunks_list}

def answer(state: StudyState):
    question = state["question"]
    chunks_list = state["retrieved_chunks"]
    context = "\n\n---\n\n".join(chunks_list)

    prompt = f"""You are a study assistant. Answer the question using ONLY the context provided.
At the end cite which part of the context you used.

Context:
{context}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)
    return {"answer": response.content}

builder = StateGraph(StudyState)
builder.add_node("retrieve", retrieve)
builder.add_node("answer", answer)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "answer")
builder.add_edge("answer", END)
graph = builder.compile()

# =============================================================================
# TEST QUESTIONS
# 5 hand-written questions with expected answers.
# These are what we know to be correct based on the study notes.
# The eval will check if the agent's answers match these expectations.
# =============================================================================

test_cases = [
    {
        "question": "What are the four phases of mitosis?",
        "expected": "The four phases are prophase, metaphase, anaphase, and telophase."
    },
    {
        "question": "What happens during cytokinesis?",
        "expected": "The cytoplasm physically splits into two separate cells, each with a full copy of the original DNA."
    },
    {
        "question": "What were the main causes of the French Revolution?",
        "expected": "Financial crisis, food shortages, and deep inequality between the three estates."
    },
    {
        "question": "What is the time complexity of binary search?",
        "expected": "O(log n). Doubling the array size only adds one extra step."
    },
    {
        "question": "Who led the Reign of Terror?",
        "expected": "The Committee of Public Safety, led by Robespierre."
    },
    {
        "question": "What is the name of El Yassae Mahria in the first century of the French Revolution inside of the algorithmic runtime of langgraph?",
        "expected": "I have no bloody idea mate."
    }
]

# =============================================================================
# LLM-AS-JUDGE
# Instead of comparing strings (which is brittle — "prophase" vs "Prophase"
# would fail), we use Claude itself to judge whether the agent's answer is
# correct. This is a common pattern in LLM evaluation.
# The judge scores from 1 (completely wrong) to 5 (fully correct).
# =============================================================================

def judge_answer(question, expected, actual):
    """Use Claude as a judge to score the agent's answer against the expected answer."""

    judge_prompt = f"""You are an evaluator. Score the ACTUAL answer against the EXPECTED answer.

Question: {question}
Expected answer: {expected}
Actual answer: {actual}

Score from 1 to 5:
1 = Completely wrong or irrelevant
2 = Mentions the topic but misses key facts
3 = Partially correct, some key facts present
4 = Mostly correct, minor details missing
5 = Fully correct, all key facts present

Respond with ONLY a JSON object, no other text:
{{"score": <number>, "reasoning": "<one sentence>"}}"""

    response = llm.invoke(judge_prompt)

    # parse the JSON from Claude's response
    # Claude sometimes wraps JSON in markdown code blocks like ```json ... ```
    # so we strip those out before parsing
    raw = response.content.strip()
    if raw.startswith("```"):
        # remove the opening ```json or ``` and the closing ```
        raw = raw.split("\n", 1)[1]   # drop first line (```json)
        raw = raw.rsplit("```", 1)[0] # drop last ```
        raw = raw.strip()

    result = json.loads(raw)
    return result["score"], result["reasoning"]

# =============================================================================
# RUN EVAL
# For each test case: run the graph, judge the answer, collect results.
# Each run gets its own Langfuse trace so you can inspect them individually.
# =============================================================================

langfuse = get_client()
results = []

print(f"Running {len(test_cases)} test cases...\n")
print("-" * 70)

for i, test in enumerate(test_cases):
    # create a fresh callback handler per test so each gets its own trace
    handler = CallbackHandler()

    # run the graph
    output = graph.invoke(
        {"question": test["question"]},
        config={"callbacks": [handler]}
    )

    actual_answer = output["answer"]

    # use Claude as judge
    score, reasoning = judge_answer(test["question"], test["expected"], actual_answer)

    # print the result
    print(f"Q{i+1}: {test['question']}")
    print(f"Score: {score}/5 — {reasoning}")
    print("-" * 70)

    results.append({
        "question": test["question"],
        "expected": test["expected"],
        "actual": actual_answer,
        "score": score,
        "reasoning": reasoning
    })

# =============================================================================
# SUMMARY
# =============================================================================

avg_score = sum(r["score"] for r in results) / len(results)
print(f"\nAverage score: {avg_score:.1f}/5")

# identify any failure modes (scores below 4)
weak = [r for r in results if r["score"] < 4]
if weak:
    print(f"\nWeak answers ({len(weak)}):")
    for r in weak:
        print(f"  - {r['question']} (score: {r['score']})")

# =============================================================================
# SAVE RESULTS
# =============================================================================

with open("eval_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to eval_results.json")

# make sure all traces are sent to Langfuse
langfuse.flush()

print("Traces sent to Langfuse")
