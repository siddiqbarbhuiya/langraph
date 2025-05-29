# agents.py
import operator
from typing import TypedDict, Annotated, List, Dict, Any

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

from chroma_utils import ChromaDBManager

# Initialize ChromaDBManager globally for agents to use
# This ensures agents share the same DB instance
db_manager = ChromaDBManager()

# --- Tools for Agents ---
@tool
def search_resumes_tool(query: str) -> List[Dict[str, Any]]:
    """
    Searches the resume database for relevant resumes based on a natural language query.
    Returns a list of dictionaries, where each dictionary contains 'name', 'role',
    'experience', and a 'content_snippet' of the found resume.
    """
    print(f"Tool: search_resumes_tool called with query: '{query}'")
    results = db_manager.search_resumes(query, k=3) # Retrieve top 3 for agents
    formatted_results = []
    for doc in results:
        formatted_results.append({
            "name": doc.metadata.get("name", "N/A"),
            "role": doc.metadata.get("role", "N/A"),
            "experience": doc.metadata.get("experience", "N/A"),
            # "content_snippet": doc.page_content[:200] + "..." # Limit snippet length
            "content_snippet": doc.page_content
        })
    return formatted_results

# --- Single Agent Setup ---
def create_single_resume_agent(llm: ChatOpenAI):
    """
    Creates a single LangChain agent for conversational resume search.
    This agent can use the search_resumes_tool.
    """
    tools = [search_resumes_tool]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an AI assistant specialized in finding resumes. "
                         "Use the 'search_resumes_tool' to find relevant resumes based on the user's request. "
                         "Be helpful and conversational, summarizing the key details of the found resumes. "
                         "If no resumes are found, politely inform the user."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    # create_openai_functions_agent is suitable for models that support function calling
    agent = create_openai_functions_agent(llm, tools, prompt)
    # AgentExecutor runs the agent, handling tool calls and response generation
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor

# --- Multi-Agent Setup with LangGraph ---

# Define the state for the LangGraph
class AgentState(TypedDict):
    """
    Represents the state of our multi-agent system.
    - input: The user's initial query.
    - chat_history: A list of messages in the conversation.
    - search_query: The extracted query for resume search.
    - search_results: Results from the resume search tool.
    - final_response: The final response to the user.
    """
    input: str
    chat_history: Annotated[List[BaseMessage], operator.add] # Accumulate chat history
    search_query: str # Extracted query from user input
    search_results: List[Dict[str, Any]] # Results from the search tool
    final_response: str # The response to be sent back to the user

# Define nodes for the graph
def query_understanding_node(state: AgentState):
    """
    Node for understanding the user's query and extracting precise search terms.
    Uses an LLM to refine the input into a search query.
    """
    print("Multi-Agent: Query Understanding Node")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) # Use a cheaper model for this
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at extracting precise search queries for a resume database. "
                     "Given the user's request, identify the most relevant terms and requirements for a resume search. "
                     "Focus on job roles, experience, and key skills. "
                     "Respond ONLY with the extracted search query string, nothing else. "
                     "Example: 'I need a Python developer with Django experience' -> 'Python developer with Django experience'\n"
                     "Example: 'Do you have any senior full stack engineers?' -> 'senior full stack engineer'\n"
                     "Example: 'What about someone with DevOps experience?' -> 'DevOps engineer'"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    chain = prompt | llm
    # Invoke the LLM to get the extracted search query
    response = chain.invoke({"input": state["input"], "chat_history": state["chat_history"]})
    extracted_query = response.content.strip()
    print(f"Extracted Search Query: {extracted_query}")
    return {"search_query": extracted_query}

def resume_search_node(state: AgentState):
    """
    Node for performing the resume search using the extracted query.
    Calls the 'search_resumes_tool'.
    """
    print("Multi-Agent: Resume Search Node")
    search_query = state["search_query"]
    # Invoke the tool with the extracted query
    results = search_resumes_tool.invoke({"query": search_query})
    print(f"Search Results: {results}")
    return {"search_results": results}

# def response_generation_node(state: AgentState): # Agent_State -> AgentState fix confirmed here
#     """
#     Node for generating a conversational response based on search results.
#     Uses an LLM to synthesize the results into a user-friendly message.
#     """
#     print("Multi-Agent: Response Generation Node")
#     llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7) # Allow some creativity for response generation
#     search_results = state["search_results"]
#     original_input = state["input"]
#     chat_history = state["chat_history"]

#     if not search_results:
#         response_content = "I couldn't find any resumes matching your request. Could you please try rephrasing or providing more details?"
#     else:
#         # Format results for the LLM prompt
#         results_str = "\n".join([
#             f"- {res['name']} ({res['role']}, {res['experience']}): {res['content_snippet']}"
#             for res in search_results
#         ])

#         # Construct messages for the prompt, including tool results as a HumanMessage
#         messages_for_llm = [
#             ("system", "You are a helpful AI assistant for resume searching. "
#                          "Based on the user's request and the provided resume search results, "
#                          "generate a polite and informative conversational response. "
#                          "Summarize the key findings from the resumes. "
#                          "If no resumes are found, inform the user."),
#             MessagesPlaceholder(variable_name="chat_history"),
#             ("human", "{input}"),
#             # Integrate tool results as a HumanMessage. This is the fix for the "Unexpected message type" error.
#             HumanMessage(content=f"Here are the resume search results:\n{results_str}"),
#             ("human", "Now, please provide a conversational summary of these results based on my original request."),
#         ]

#         prompt = ChatPromptTemplate.from_messages(messages_for_llm) # Use the corrected messages list
#         chain = prompt | llm
#         # Invoke the LLM to generate the final response
#         response = chain.invoke({
#             "input": original_input,
#             "chat_history": chat_history,
#             # 'tool_results' is now included in the prompt's messages, so no need to pass it separately here
#         })
#         response_content = response.content.strip()
#     print(f"Generated Response: {response_content}")
#     return {"final_response": response_content}


def response_generation_node(state: AgentState):
    """
    Node for generating a conversational response based on search results.
    Uses an LLM to synthesize the results into a user-friendly message.
    """
    print("Multi-Agent: Response Generation Node")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    search_results = state["search_results"]
    original_input = state["input"]
    chat_history = state["chat_history"]

    if not search_results:
        response_content = "I couldn't find any resumes matching your request. Could you please try rephrasing or providing more details?"
    else:
        # Format results for the LLM prompt
        results_str = "\n".join([
            f"- {res['name']} ({res['role']}, {res['experience']}): {res['content_snippet']}"
            for res in search_results
        ])

        # --- MODIFIED SYSTEM PROMPT HERE ---
        messages_for_llm = [
            ("system", "You are a helpful AI assistant for resume searching. "
                       "Based on the user's request and the provided resume search results, "
                       "generate a polite and informative conversational response. "
                       "If the user asks for a *specific detail* (like 'academic details', 'certifications', 'company experience') "
                       "for a particular candidate, **extract ONLY that specific detail** from the provided resume content. "
                       "If no specific detail is requested, summarize the key findings. "
                       "If no resumes are found, inform the user."), # Added clear instruction for specific details
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            HumanMessage(content=f"Here are the resume search results:\n{results_str}"),
            ("human", "Now, please provide a conversational response based on my original request, being precise if a specific detail was asked."),
        ]

        prompt = ChatPromptTemplate.from_messages(messages_for_llm)
        chain = prompt | llm
        response = chain.invoke({
            "input": original_input,
            "chat_history": chat_history,
        })
        response_content = response.content.strip()
    print(f"Generated Response: {response_content}")
    return {"final_response": response_content}

def create_multi_agent_resume_searcher():
    """
    Creates a multi-agent system using LangGraph for conversational resume search.
    Defines the workflow and transitions between different agent nodes.
    """
    workflow = StateGraph(AgentState)

    # Add nodes to the workflow
    workflow.add_node("query_understanding", query_understanding_node)
    workflow.add_node("resume_search", resume_search_node)
    workflow.add_node("response_generation", response_generation_node)

    # Define the entry point and edges (transitions) between nodes
    workflow.set_entry_point("query_understanding") # Start here
    workflow.add_edge("query_understanding", "resume_search") # After understanding, search
    workflow.add_edge("resume_search", "response_generation") # After searching, generate response
    workflow.add_edge("response_generation", END) # The response generation is the end of the current turn

    # Compile the workflow into a runnable LangGraph application
    app = workflow.compile()
    return app