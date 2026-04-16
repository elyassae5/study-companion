import os
from dotenv import load_dotenv

# Load API keys from .env file into environment variables
load_dotenv()

# --- Check that the key was loaded ---
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise EnvironmentError("ANTHROPIC_API_KEY not found in .env — check the file exists and has the key")

print("API key loaded ok")

# --- Test all imports ---
# LangGraph is the graph execution framework we'll use to wire up nodes
from langgraph.graph import StateGraph, START, END

# LangChain is the glue layer — it gives us a standard interface to call any LLM
from langchain_anthropic import ChatAnthropic

# langfuse will be used later for tracing — just confirming it's installed
import langfuse

print("All imports ok")

# --- Make a single test call to Claude Haiku ---
# ChatAnthropic is LangChain's wrapper around the Anthropic API
# It uses ANTHROPIC_API_KEY from the environment automatically
llm = ChatAnthropic(model="claude-haiku-4-5-20251001")

response = llm.invoke("Say exactly: Environment verified, Claude responds")

print(response.content)
