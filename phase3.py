import sys
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# SETUP — rebuild the vector store (same logic as phase2)
# In a real app this would be loaded from disk, not rebuilt every run
# =============================================================================

loader = TextLoader("data/notes.md", encoding="utf-8")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

print("Vector store ready")

# LLM used by the answer node
llm = ChatAnthropic(model="claude-haiku-4-5-20251001")

# =============================================================================
# STATE SCHEMA
# TypedDict defines the shape of the shared state object that flows through
# the entire graph. Every node reads from this dict and returns a partial
# update to it — only the keys it changed, not the whole dict.
# =============================================================================

class StudyState(TypedDict):
    question: str           # the user's question — set once at the start
    retrieved_chunks: list  # populated by the retrieve node
    answer: str             # populated by the answer node

# =============================================================================
# NODE 1 — retrieve
# A node is just a Python function. It receives the current state and returns
# a dict containing only the keys it wants to update in state.
# This node reads the question, searches the vector store, writes chunks back.
# =============================================================================

def retrieve(state: StudyState):
    question = state["question"]

    # similarity_search embeds the question and finds the closest chunks
    results = vectorstore.similarity_search(question, k=3)

    # results are Document objects — extract just the text
    chunks = [doc.page_content for doc in results]

    # return only the keys we're updating — LangGraph merges this into state
    return {"retrieved_chunks": chunks}

# =============================================================================
# NODE 2 — answer
# Reads question + chunks from state, builds a grounded prompt, calls Claude,
# writes the answer back to state.
# =============================================================================

def answer(state: StudyState):
    question = state["question"]
    chunks = state["retrieved_chunks"]

    # join the chunks into a readable context block
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
# GRAPH DEFINITION
# StateGraph is the builder object — you register nodes and edges, then call
# .compile() to turn it into an executable graph.
# =============================================================================

# pass our state schema so the graph knows what shape the state is
builder = StateGraph(StudyState)

# register nodes — give each function a name inside the graph
builder.add_node("retrieve", retrieve)
builder.add_node("answer", answer)

# wire the edges — this defines the execution order
# START and END are special constants, not real nodes that do work
builder.add_edge(START, "retrieve")    # graph enters here
builder.add_edge("retrieve", "answer") # retrieve runs, then answer
builder.add_edge("answer", END)        # answer is the last step

# compile() locks the graph and makes it executable
# before compile() it's just a definition — after, it's a runnable pipeline
graph = builder.compile()

print("Graph compiled\n")

# =============================================================================
# RUN
# Pass the initial state — only question is set, the rest start empty.
# LangGraph runs retrieve first, which fills retrieved_chunks,
# then runs answer, which fills answer.
# invoke() returns the final state after all nodes have run.
# =============================================================================

question = sys.argv[1] if len(sys.argv) > 1 else "What are the phases of mitosis?"

print(f"Question: {question}\n")

result = graph.invoke({"question": question})

print(f"Answer:\n{result['answer']}")
