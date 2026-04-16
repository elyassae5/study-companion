# Study Companion

A LangGraph-based study assistant that answers questions grounded in your study notes using RAG (Retrieval-Augmented Generation). It routes questions through a conditional graph — retrieval questions search your notes, general questions get answered directly.

## How it works

```
                         ┌─────────────────────────────────┐
                         │          START                   │
                         └────────────┬────────────────────┘
                                      │
                                      v
                              ┌───────────────┐
                              │    router      │
                              │ (classifies    │
                              │  the question) │
                              └───────┬───────┘
                                      │
                          ┌───────────┴───────────┐
                          │ pick_route()          │
                          │ "retrieval"     "clarification"
                          v                       v
                   ┌────────────┐         ┌──────────────┐
                   │  retrieve  │         │ direct_answer │
                   │ (search    │         │ (answer       │
                   │  notes)    │         │  directly)    │
                   └─────┬──────┘         └──────┬───────┘
                         │                       │
                         v                       │
                   ┌────────────┐                │
                   │   answer   │                │
                   │ (grounded  │                │
                   │  response) │                │
                   └─────┬──────┘                │
                         │                       │
                         v                       v
                         ┌─────────────────────────┐
                         │           END            │
                         └─────────────────────────┘
```

## Setup

```bash
python -m venv venv
source venv/Scripts/activate   # Git Bash on Windows
pip install -r requirements.txt
```

Create a `.env` file:

```
ANTHROPIC_API_KEY=sk-ant-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

## Usage

Ask a question:

```bash
python phase6.py "What are the phases of mitosis?"
```

Run the eval suite:

```bash
python eval.py
```

## Eval results

| Question | Score |
|----------|-------|
| What are the four phases of mitosis? | 5/5 |
| What happens during cytokinesis? | 5/5 |
| What were the main causes of the French Revolution? | 5/5 |
| What is the time complexity of binary search? | 5/5 |
| Who led the Reign of Terror? | 5/5 |
| Nonsense question (unanswerable) | 4/5 |

**Average: 4.8/5**

## What I'd improve next

- **Persist the vector store to disk** — currently rebuilds from scratch every run, which is slow and wasteful with larger note sets
- **Add more study notes** — 3 topics is a demo; a real system would handle dozens of topics and test cross-topic questions
- **Chunk strategy tuning** — the fixed 500-char chunks work here but could miss context for longer, more nuanced topics
- **Streaming responses** — currently waits for the full answer before printing; streaming would feel more responsive
- **Better router** — the LLM-based router works but adds latency (an extra API call); a lightweight classifier or keyword matcher could route faster
- **Evaluation depth** — 5 test cases is a starting point; real eval needs edge cases, adversarial inputs, and multi-hop questions that span topics
