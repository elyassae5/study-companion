# Study Companion — Project Context

## Purpose

This project exists for learning. I am building a small LangGraph-based study companion agent to get hands-on experience with LangGraph, Langfuse, and LLM evaluation. I have an interview tomorrow for a Student AI Engineer role at a company that uses these tools daily, so I need to understand every piece of what we build, not just have working code.

## How we work together

**Phase by phase. No skipping ahead.**

After completing each phase, stop and explain:
- What we just built
- What each part does and why
- How it connects to what we built before
- What we're doing next and why

Do not move to the next phase until I say I understand and I'm ready. If I ask questions, answer them clearly before continuing. This is a learning project, not a speed run.

## The phases

### Phase 1 — Environment setup
- Verify Python version, install dependencies
- Confirm all imports work (langgraph, langchain, langchain-anthropic, langfuse)
- Load API keys from .env
- Run a single test call to Claude Haiku to confirm the Anthropic key works
- Output: "Environment verified, Claude responds"

### Phase 2 — Seed data and vector store
- Create sample study notes in data/notes.md (3 topics, ~500 words total: mitosis, the French Revolution, binary search)
- Chunk the notes (~100 words per chunk)
- Embed chunks using HuggingFace sentence-transformers locally (all-MiniLM-L6-v2) so we don't need an OpenAI key
- Store in FAISS (in-memory)
- Test: query the vector store with a question, confirm it returns relevant chunks
- Output: "Vector store works, retrieval returns relevant chunks"

### Phase 3 — Basic LangGraph flow (2 nodes)
- Define state schema as a TypedDict: question, retrieved_chunks, answer
- Build a StateGraph with two nodes:
  - `retrieve`: takes the question, queries the vector store, returns top-k chunks
  - `answer`: takes question + chunks, calls Claude Haiku, returns a grounded answer with citations
- Wire the nodes with edges: START -> retrieve -> answer -> END
- Test: run the full graph with a sample question
- Output: "LangGraph flow runs end to end, answer is grounded in chunks"

### Phase 4 — Langfuse tracing
- Integrate Langfuse CallbackHandler so every LLM call is traced
- Run a question through the graph
- Confirm traces appear in the Langfuse dashboard
- Output: "Traces visible in Langfuse, can see latency, tokens, and prompt/response"

### Phase 5 — Evaluation script
- Write eval.py with 5 hand-written test questions and expected answers
- For each: run the graph, then use Claude as LLM-judge to score the answer (1 to 5 scale)
- Print a results table to console
- Save results to eval_results.json
- Output: "Eval suite runs, scores are reasonable, at least one failure mode identified"

### Phase 6 (stretch) — Router node
- Only if Phases 1 through 5 are fully working
- Add a `router` node that classifies the question as "retrieval" (needs to search notes) vs "clarification" (can answer directly)
- Use conditional edges based on the router's output
- Output: "3-node graph with conditional routing"

### Phase 7 (stretch) — README
- What it does (2 sentences)
- ASCII flow diagram of the graph
- How to run it
- Eval results summary
- Honest list of what I'd improve next

## Technical constraints

- **LLM: Anthropic only.** Use claude-haiku-4-5 via langchain-anthropic. I have the API key, no OpenAI key.
- **Embeddings: local HuggingFace.** Use sentence-transformers/all-MiniLM-L6-v2 via langchain-community. No OpenAI embeddings.
- **Vector store: FAISS in-memory.** No external DB needed for this project.
- **No LangChain prebuilt agents.** Build the LangGraph StateGraph manually with explicit nodes and edges. I need to understand the internals, not use abstractions.
- **Comment heavily.** Every LangGraph concept (StateGraph, nodes, edges, conditional edges, state schema) should have a comment explaining what it is and why it's there. I need to explain these verbally in an interview.
- **Single entry point.** `python main.py "What is mitosis?"` should work.
- **Keep it simple.** One file per phase is fine. Don't create unnecessary abstractions, classes, or layers. Flat and readable beats elegant and confusing.

## Tech stack

- Python 3.12
- langgraph
- langchain, langchain-anthropic, langchain-community
- sentence-transformers (all-MiniLM-L6-v2)
- faiss-cpu
- langfuse (with langfuse callback for langchain)
- python-dotenv

## What I already know well

- Python (FastAPI, async, scripting)
- Anthropic SDK (used directly in other projects)
- RAG concepts (built a production RAG system with LlamaIndex + Pinecone)
- LLM-as-judge (used in a hackathon with Google Gemini)
- n8n, Docker, VPS deployment

## What I am learning through this project

- LangGraph (StateGraph, nodes, edges, conditional routing, state management)
- Langfuse (tracing, observability, monitoring LLM calls)
- LangChain integration patterns (ChatAnthropic, HuggingFaceEmbeddings, FAISS)
- Structured LLM evaluation with scoring logic