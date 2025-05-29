# main.py
import os
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Check for OpenAI API Key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in your .env file.")

# Initialize LangChain LLM (using gpt-4o-mini for cost-effectiveness)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Initialize ChromaDB Manager and load data
from chroma_utils import ChromaDBManager
from resume_data import RESUMES

db_manager = ChromaDBManager()

# Ensure ChromaDB is populated on application startup
# In a production environment, this data ingestion would typically be a separate,
# more robust process (e.g., a dedicated script, a data pipeline).
# Here, we check if the collection is empty and populate it if needed.
try:
    # Attempt to count documents in the collection
    if db_manager.collection.count() == 0:
        print("ChromaDB collection is empty. Populating with sample resumes...")
        db_manager.add_resumes(RESUMES)
    else:
        print(f"ChromaDB collection already contains {db_manager.collection.count()} documents.")
except Exception as e:
    print(f"Error checking/populating ChromaDB: {e}. "
          "If this is the first run, ensure 'chroma_utils.py' was run directly once "
          "or handle initial population more robustly in production.")
    # As a fallback, try adding data, but be aware of potential duplicates if run repeatedly
    # without proper collection clearing.
    db_manager.add_resumes(RESUMES)
    


# Initialize LangChain Agents
from agents import create_single_resume_agent, create_multi_agent_resume_searcher

# Create instances of our agents
single_agent_executor = create_single_resume_agent(llm)
multi_agent_app = create_multi_agent_resume_searcher()

# Initialize FastAPI application
app = FastAPI(
    title="AI Resume Searcher",
    description="API for searching resumes and interacting via chat using AI agents. "
                "Supports both direct search and conversational queries with single and multi-agent architectures.",
    version="1.0.0"
)

# --- Pydantic Models for API Request/Response ---

class SearchQuery(BaseModel):
    """Request model for direct resume search."""
    query: str

class ResumeResult(BaseModel):
    """Response model for a single resume result."""
    name: str
    role: str
    experience: str
    content_snippet: str

class SearchResponse(BaseModel):
    """Overall response model for direct resume search."""
    results: List[ResumeResult]
    message: str = "Search completed successfully."

class ChatMessage(BaseModel):
    """Represents a single message in the chat history."""
    role: str # "human" or "ai"
    content: str

class ChatRequest(BaseModel):
    """Request model for the conversational chat feature."""
    user_message: str
    chat_history: List[ChatMessage] = [] # Optional: previous messages for context
    agent_type: str = "single" # "single" or "multi" to choose agent architecture

class ChatResponse(BaseModel):
    """Response model for the conversational chat feature."""
    response: str
    updated_chat_history: List[ChatMessage]


# --- API Endpoints ---

@app.post("/search_resumes", response_model=SearchResponse, summary="Direct Resume Search")
async def search_resumes(search_query: SearchQuery):
    """
    **Directly searches for resumes based on a given query string.**

    This endpoint uses semantic similarity to find resumes that best match the `query`.
    It's ideal for immediate, non-conversational searches.

    - **`query`**: The search string (e.g., "Python developer with Django experience", "Data Scientist specializing in NLP").
    """
    try:
        # Perform the search using the ChromaDBManager
        results = db_manager.search_resumes(search_query.query, k=5) # Retrieve top 5 for direct search
        formatted_results = []
        for doc in results:
            formatted_results.append(ResumeResult(
                name=doc.metadata.get("name", "N/A"),
                role=doc.metadata.get("role", "N/A"),
                experience=doc.metadata.get("experience", "N/A"),
                # content_snippet=doc.page_content[:200] + "..." # Limit snippet length for response
                content_snippet=doc.page_content
            ))
        return SearchResponse(results=formatted_results)
    except Exception as e:
        # Catch any exceptions during the search process and return a 500 error
        raise HTTPException(status_code=500, detail=f"Error during resume search: {str(e)}")

@app.post("/chat_resumes", response_model=ChatResponse, summary="Conversational Resume Search")
async def chat_resumes(request: ChatRequest):
    """
    **Engages in a conversational chat to find resumes using AI agents.**

    This endpoint allows for natural language interaction to query resumes.
    You can specify whether to use a `single` agent (simpler, direct tool use)
    or a `multi` agent (LangGraph-based, more structured workflow).

    - **`user_message`**: The current message from the user.
    - **`chat_history`**: (Optional) A list of previous `ChatMessage` objects to maintain conversation context.
    - **`agent_type`**: Specifies which agent architecture to use ("single" or "multi").
    """
    try:
        # Convert Pydantic ChatMessage objects to LangChain BaseMessage objects
        # This is necessary because LangChain agents expect their own message types.
        lc_chat_history = []
        for msg in request.chat_history:
            if msg.role == "human":
                lc_chat_history.append(HumanMessage(content=msg.content))
            elif msg.role == "ai":
                lc_chat_history.append(AIMessage(content=msg.content))
            # Ignore other roles or handle as needed

        ai_response_content: str

        if request.agent_type == "single":
            print("Using Single Agent for chat.")
            # Invoke the single agent with the current user message and chat history
            response = await single_agent_executor.ainvoke({
                "input": request.user_message,
                "chat_history": lc_chat_history
            })
            ai_response_content = response["output"] # The output from the single agent
        elif request.agent_type == "multi":
            print("Using Multi-Agent (LangGraph) for chat.")
            # LangGraph app expects an initial state dictionary.
            # It will process through its nodes and return the final state.
            final_state = await multi_agent_app.ainvoke({
                "input": request.user_message,
                "chat_history": lc_chat_history,
                "search_query": "", # Initialize empty, will be populated by query_understanding_node
                "search_results": [], # Initialize empty, will be populated by resume_search_node
                "final_response": "" # Initialize empty, will be populated by response_generation_node
            })
            ai_response_content = final_state["final_response"] # The final response from the multi-agent graph
        else:
            # Handle invalid agent_type gracefully
            raise HTTPException(status_code=400, detail="Invalid agent_type. Must be 'single' or 'multi'.")

        # Update chat history for the response
        # Add the current human message and the AI's response to the history
        updated_lc_chat_history = lc_chat_history + [
            HumanMessage(content=request.user_message),
            AIMessage(content=ai_response_content)
        ]
        # Convert the updated LangChain history back to Pydantic ChatMessage for the API response
        updated_pydantic_chat_history = [
            ChatMessage(role="human", content=msg.content) if isinstance(msg, HumanMessage)
            else ChatMessage(role="ai", content=msg.content)
            for msg in updated_lc_chat_history
        ]

        return ChatResponse(
            response=ai_response_content,
            updated_chat_history=updated_pydantic_chat_history
        )
    except Exception as e:
        print(f"Error during chat: {e}") # Log the error for debugging
        raise HTTPException(status_code=500, detail=f"Error during chat: {str(e)}")

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint, redirects to API documentation."""
    return {"message": "Welcome to the AI Resume Searcher API. Visit /docs for API documentation."}