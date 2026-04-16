import sys
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

# --- NEW IN PHASE 4 ---
# Langfuse is an observability tool — it records every LLM call so you can
# see exactly what prompt went in, what response came out, how long it took,
# and how many tokens were used. Think of it like browser DevTools but for LLM calls.
from langfuse import get_client
from langfuse.langchain import CallbackHandler

load_dotenv()

# =============================================================================
# SETUP — same vector store as before
# =============================================================================

loader = TextLoader("data/notes.md", encoding="utf-8")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

print("Vector store ready")

llm = ChatAnthropic(model="claude-haiku-4-5-20251001")

# --- NEW IN PHASE 4 ---
# CallbackHandler is the bridge between LangChain and Langfuse.
# It automatically reads LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, and
# LANGFUSE_BASE_URL from your environment variables (loaded by dotenv above).
# No constructor arguments needed — it configures itself from the env vars.
langfuse_handler = CallbackHandler()

print("Langfuse handler ready")

# =============================================================================
# STATE + NODES — same as phase3
# =============================================================================

class StudyState(TypedDict):
    question: str
    retrieved_chunks: list
    answer: str

def retrieve(state: StudyState):
    question = state["question"]
    results = vectorstore.similarity_search(question, k=3)
    chunks = [doc.page_content for doc in results]
    return {"retrieved_chunks": chunks}

def answer(state: StudyState):
    question = state["question"]
    chunks = state["retrieved_chunks"]
    context = "\n\n---\n\n".join(chunks)

    prompt = f"""You are a study assistant. Answer the question using ONLY the context provided.
At the end cite which part of the context you used.

Context:
{context}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)
    return {"answer": response.content}

# =============================================================================
# GRAPH — same wiring as phase3
# =============================================================================

builder = StateGraph(StudyState)
builder.add_node("retrieve", retrieve)
builder.add_node("answer", answer)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "answer")
builder.add_edge("answer", END)
graph = builder.compile()

print("Graph compiled\n")

# =============================================================================
# RUN — the only change is passing the langfuse callback
# =============================================================================

question = sys.argv[1] if len(sys.argv) > 1 else "What are the phases of mitosis?"

print(f"Question: {question}\n")

# --- NEW IN PHASE 4 ---
# config={"callbacks": [langfuse_handler]} is the key line.
# This tells LangGraph to pass the Langfuse handler to every LLM call
# inside the graph. Langfuse then records the full trace automatically —
# what prompt went in, what came out, latency, token count, everything.
result = graph.invoke(
    {"question": question},
    config={"callbacks": [langfuse_handler]}
)

print(f"Answer:\n{result['answer']}")

# --- NEW IN PHASE 4 ---
# flush() ensures all traces are sent to Langfuse before the script exits.
# Without this, the script might exit before the traces finish uploading
# (they're sent asynchronously in the background).
# get_client() returns the singleton Langfuse client that the handler uses internally.
get_client().flush()

print("\nTraces sent to Langfuse — check your dashboard!")
