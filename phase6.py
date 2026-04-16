import sys
from typing import Literal
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
# SETUP — same as before
# =============================================================================

loader = TextLoader("data/notes.md", encoding="utf-8")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

llm = ChatAnthropic(model="claude-haiku-4-5-20251001")

print("Setup complete")

# =============================================================================
# STATE SCHEMA
# Same as before, but we add a "route" field so the router node can tell
# the graph which path to take.
# =============================================================================

class StudyState(TypedDict):
    question: str
    route: str              # NEW — "retrieval" or "clarification"
    retrieved_chunks: list
    answer: str

# =============================================================================
# NODE 1 — router (NEW)
# This node classifies the question BEFORE doing any work.
# "retrieval" = the question needs to search the notes for specific facts
# "clarification" = the question is general and can be answered directly
# The router uses Claude to decide — it reads the question and picks a route.
# =============================================================================

def router(state: StudyState):
    question = state["question"]

    prompt = f"""Classify this question into exactly one category.

"retrieval" — the question asks about a specific topic, fact, or concept that
would need to be looked up in study notes (e.g. mitosis, French Revolution,
binary search, historical events, scientific processes, algorithms).

"clarification" — the question is general, conversational, or asks for
explanation of a broad concept that doesn't need specific notes
(e.g. "what does studying mean?", "how should I prepare for exams?",
"can you help me?").

Question: {question}

Respond with ONLY one word: retrieval or clarification"""

    response = llm.invoke(prompt)
    route = response.content.strip().lower()

    # make sure it's one of the two valid values
    if route not in ("retrieval", "clarification"):
        route = "retrieval"  # default to retrieval if unclear

    print(f"Router decided: {route}")
    return {"route": route}

# =============================================================================
# NODE 2 — retrieve (same as before)
# =============================================================================

def retrieve(state: StudyState):
    question = state["question"]
    results = vectorstore.similarity_search(question, k=3)
    chunks_list = [doc.page_content for doc in results]
    return {"retrieved_chunks": chunks_list}

# =============================================================================
# NODE 3 — answer (same as before, uses retrieved chunks)
# =============================================================================

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

# =============================================================================
# NODE 4 — direct_answer (NEW)
# This node handles "clarification" questions — ones that don't need the
# vector store. It answers directly without any retrieved context.
# =============================================================================

def direct_answer(state: StudyState):
    question = state["question"]

    prompt = f"""You are a study assistant. Answer this general question helpfully and concisely.

Question: {question}

Answer:"""

    response = llm.invoke(prompt)
    return {"answer": response.content}

# =============================================================================
# ROUTING FUNCTION
# This function is NOT a node — it's a function that reads the state and
# returns a string telling LangGraph which node to go to next.
# It runs AFTER the router node and uses the "route" field that router set.
# =============================================================================

def pick_route(state: StudyState) -> Literal["retrieve", "direct_answer"]:
    if state["route"] == "clarification":
        return "direct_answer"
    return "retrieve"

# =============================================================================
# GRAPH DEFINITION
# Before (phase4):  START → retrieve → answer → END  (straight line)
# Now (phase6):     START → router → (retrieve → answer) OR (direct_answer) → END
#
# The conditional edge after "router" is the key difference.
# Instead of add_edge (which always goes to one place), we use
# add_conditional_edges which calls pick_route() to decide where to go.
# =============================================================================

builder = StateGraph(StudyState)

# register all nodes
builder.add_node("router", router)
builder.add_node("retrieve", retrieve)
builder.add_node("answer", answer)
builder.add_node("direct_answer", direct_answer)

# wire the edges
builder.add_edge(START, "router")              # always start at router

# conditional edge — after router, call pick_route() to decide next node
builder.add_conditional_edges("router", pick_route)

builder.add_edge("retrieve", "answer")         # if retrieval path: retrieve → answer
builder.add_edge("answer", END)                # answer exits the graph
builder.add_edge("direct_answer", END)         # direct_answer also exits the graph

graph = builder.compile()

print("Graph compiled\n")

# =============================================================================
# RUN
# =============================================================================

question = sys.argv[1] if len(sys.argv) > 1 else "What are the phases of mitosis?"

langfuse_handler = CallbackHandler()

print(f"Question: {question}\n")

result = graph.invoke(
    {"question": question},
    config={"callbacks": [langfuse_handler]}
)

print(f"Answer:\n{result['answer']}")

get_client().flush()
