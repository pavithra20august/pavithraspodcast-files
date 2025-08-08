# AI Agent with MCP and LangGraph

This is a simple AI Agent that demonstrates the use of **MCP (Model Context Protocol)** and **LangGraph** for building intelligent workflows.

## Architecture

### MCP (Model Context Protocol)
- **MCPClient Class**: Provides a standardized interface for model communication
- **Model-agnostic**: Can easily switch between different LLM providers (Gemini, OpenAI, etc.)
- **Standardized messaging**: Uses LangChain message types for consistent communication

### LangGraph Workflow
The agent uses a 3-node LangGraph workflow:

1. **process_question**: Validates and formats the user's question
2. **generate_answer**: Uses MCP client to get LLM response
3. **format_response**: Formats the final answer for display

## Code Structure

### MCP Implementation
```python
class MCPClient:
    def generate_response(self, messages: list) -> str:
        # Standardized model communication
```

### LangGraph Implementation
```python
def build_agent_graph():
    workflow = StateGraph(AgentState)
    # Add nodes and edges
    return workflow.compile()
```

## Usage

1. **Start the server**: `uvicorn main:app --reload`
2. **Access the UI**: Go to `http://localhost:8000`
3. **Ask questions**: The agent will process them through the MCP + LangGraph workflow

## Debug Endpoint

Visit `http://localhost:8000/debug/{your_question}` to see the workflow steps and state changes.

## State Management

The `AgentState` TypedDict tracks:
- `messages`: Conversation history
- `question`: Current user question
- `answer`: Generated response
- `processing_step`: Current workflow step

## Benefits

- **Modular**: Easy to add new nodes or modify workflow
- **Extensible**: Can add more complex reasoning steps
- **Model-agnostic**: MCP allows switching LLM providers
- **Observable**: Debug endpoint shows workflow execution 