from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Step 1: Load the notes file ---
# TextLoader reads a file from disk and returns it as a LangChain Document object
# A Document is just a wrapper: it holds the text content + some metadata (like filename)
loader = TextLoader("data/notes.md", encoding="utf-8")
documents = loader.load()

print(f"Loaded {len(documents)} document(s)")

# --- Step 2: Chunk the documents ---
# LLMs have context limits and vector search works better on small focused chunks
# RecursiveCharacterTextSplitter splits text by trying "\n\n", then "\n", then " "
# It tries to keep splits at natural boundaries (paragraphs first, then lines, then words)
# chunk_size=500 means each chunk is at most 500 characters
# chunk_overlap=50 means consecutive chunks share 50 characters — prevents losing context at boundaries
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

print(f"Split into {len(chunks)} chunks")
print(f"Example chunk:\n{chunks[0].page_content}\n")

# --- Step 3: Embed the chunks ---
# Embeddings convert text into a list of numbers (a vector) that captures semantic meaning
# Similar meaning = vectors that are close together in space
# We use a local HuggingFace model so we don't need an OpenAI key
# all-MiniLM-L6-v2 is small, fast, and good enough for this use case
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("Embedding model loaded")

# --- Step 4: Store chunks in FAISS ---
# FAISS is a vector store — it stores all the chunk embeddings in memory
# and lets you search them by semantic similarity
# from_documents() embeds every chunk and indexes them all in one call
vectorstore = FAISS.from_documents(chunks, embeddings)

print("Vector store built")

# --- Step 5: Test retrieval ---
# similarity_search() embeds the query using the same model, then finds the
# closest chunk vectors in the store — "closest" means most semantically similar
query = "What are the phases of mitosis?"
results = vectorstore.similarity_search(query, k=2)

print(f"\nQuery: {query}")
print(f"Top {len(results)} results:\n")
for i, result in enumerate(results):
    print(f"--- Result {i+1} ---")
    print(result.page_content)
    print()
