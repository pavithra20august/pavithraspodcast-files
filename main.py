import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph import START
from langgraph.graph import add_messages
import json

# Load environment variables from .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str

# MCP-inspired state structure
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question: str
    answer: str
    processing_step: str

# MCP Client (simplified implementation)
class MCPClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    def generate_response(self, messages: list) -> str:
        """MCP-style standardized model communication"""
        # Convert messages to Gemini format
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, 'content'):
                prompt = last_message.content
            else:
                prompt = str(last_message)
        else:
            prompt = ""
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            return f"Error: {str(e)}"

# Initialize MCP Client
mcp_client = MCPClient(GEMINI_API_KEY)

# LangGraph Node Functions
def process_question(state: AgentState) -> AgentState:
    """LangGraph node: Process and validate the question"""
    question = state["question"]
    state["processing_step"] = "question_processed"
    state["messages"].append(HumanMessage(content=f"User question: {question}"))
    return state

def generate_answer(state: AgentState) -> AgentState:
    """LangGraph node: Generate answer using MCP client"""
    messages = state["messages"]
    answer = mcp_client.generate_response(messages)
    state["answer"] = answer
    state["processing_step"] = "answer_generated"
    state["messages"].append(AIMessage(content=answer))
    return state

def format_response(state: AgentState) -> AgentState:
    """LangGraph node: Format the final response"""
    answer = state["answer"]
    # Add some formatting/processing if needed
    formatted_answer = f"🤖 AI Agent Response:\n\n{answer}"
    state["answer"] = formatted_answer
    state["processing_step"] = "response_formatted"
    return state

# Build LangGraph workflow
def build_agent_graph():
    """Create the LangGraph workflow"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("process_question", process_question)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("format_response", format_response)
    
    # Add edges
    workflow.add_edge(START, "process_question")
    workflow.add_edge("process_question", "generate_answer")
    workflow.add_edge("generate_answer", "format_response")
    workflow.add_edge("format_response", END)
    
    return workflow.compile()

# Create the agent graph
agent_graph = build_agent_graph()

@app.get("/")
async def read_root():
    return FileResponse("frontend.html")

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """Main endpoint using MCP + LangGraph workflow"""
    try:
        # Initialize state for LangGraph
        initial_state = {
            "messages": [],
            "question": request.question,
            "answer": "",
            "processing_step": "started"
        }
        
        # Execute LangGraph workflow
        result = agent_graph.invoke(initial_state)
        
        return AskResponse(answer=result["answer"])
        
    except Exception as e:
        return AskResponse(answer=f"Error in agent workflow: {str(e)}")

# Add a debug endpoint to see the workflow steps
@app.get("/debug/{question}")
async def debug_workflow(question: str):
    """Debug endpoint to see LangGraph workflow steps"""
    initial_state = {
        "messages": [],
        "question": question,
        "answer": "",
        "processing_step": "started"
    }
    
    result = agent_graph.invoke(initial_state)
    return {
        "final_answer": result["answer"],
        "processing_steps": result["processing_step"],
        "message_count": len(result["messages"])
    } 