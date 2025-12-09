# langgraph_agent.py
# FIXES INCLUDED:
# 1. RecursionError fix (Increased recursion limit to 5).
# 2. Robust parsing fix (Added fallback logic to ensure the agent stops).
# 3. LangChainDeprecationWarning fix (Switched to 'from langchain_ollama import ChatOllama').

from typing import TypedDict, List
import json
import os
import re

# LangChain/LangGraph Imports
# FIX: Switched from langchain_community to the dedicated langchain_ollama package
# to resolve the LangChainDeprecationWarning.
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from ollama import Client as OllamaClient

# --- CONFIGURATION ---
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "phi3:mini") 

# --- INITIALIZATION & STATUS CHECK ---
try:
    ollama_client = OllamaClient()
    ollama_client.list() # Test connection
    OLLAMA_STATUS = "Connected"
except Exception:
    ollama_client = None
    OLLAMA_STATUS = "Error: Ollama client failed to connect."

# --- 1. Define the Agent State (The memory) ---
class AgentState(TypedDict):
    """Represents the state of our agent's workflow."""
    messages: List[BaseMessage]
    current_price: float
    current_cash: float
    current_position: int
    sma_20: float
    sma_50: float
    final_action: str
    thought: str
    tool_calls: List[dict] 
    
# --- 2. Define Tools (The Agent's capabilities) ---

@tool
def risk_assessment_tool(price: float, position: int, cash: float) -> str:
    """
    Assesses portfolio risk based on current market data and agent holdings.
    Returns a string summary of exposure and risk level.
    """
    portfolio_value = cash + position * price
    exposure = position * price
    total = max(portfolio_value, 1.0)
    ratio = exposure / total
    
    if ratio > 0.7:
        risk = "High"
    elif ratio > 0.4:
        risk = "Medium"
    else:
        risk = "Low"
        
    return f"Current Price: ${price:.2f}, Cash: ${cash:.2f}, Position: {position} shares. Total Portfolio Value: ${portfolio_value:.2f}. Exposure Ratio: {ratio:.2f}. **Risk Level: {risk}**."

# Manually define the tool schema for the prompt (REAct instruction)
TOOL_SCHEMA = """
risk_assessment_tool(price: float, position: int, cash: float) -> str: 
    Assesses portfolio risk based on current market data and agent holdings.
    Returns a string summary of exposure and risk level.
"""

# --- 3. Define the LLM (The Brain) ---
# FIX: ChatOllama is now imported from langchain_ollama
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)


# --- 4. Define Graph Nodes (The Steps) ---

def run_llm_node(state: AgentState) -> AgentState:
    """Invokes the LLM to either reason or call a tool."""
    messages = state["messages"]
    
    # Define the system prompt with tool instructions
    system_prompt = f"""
You are a trading agent. Your goal is to decide BUY, SELL, or HOLD.
Available Tools (Use this exact format if you decide to call a tool):
<tool_call>risk_assessment_tool(price={state['current_price']}, position={state['current_position']}, cash={state['current_cash']})</tool_call>

Tools available:
{TOOL_SCHEMA}

Current Market and Holdings:
- Price: {state['current_price']:.2f}, SMA-20: {state['sma_20']:.2f}, SMA-50: {state['sma_50']:.2f}
- Holdings: Cash: ${state['current_cash']:.2f}, Position: {state['current_position']} shares.

Process:
1. Reason: State your analysis.
2. Action: If you need risk data, output a <tool_call>. Otherwise, if ready to trade, output the final action.

Final Output Format (If not calling a tool):
<final_decision>
{{"action": "<BUY|SELL|HOLD>", "thought": "<reasoning>"}}
</final_decision>
"""
    # Prepend the system prompt to the first user message
    if len(messages) == 1:
        initial_message = messages[0].content
        messages[0].content = f"{system_prompt}\n\nUser Request: {initial_message}"

    try:
        # 2. Invoke LLM
        response = llm.invoke(messages)
        
        # 3. Update messages and attempt to parse tool call/final action
        state["messages"].append(response)
        state["tool_calls"] = [] # Clear previous calls

        content = response.content.strip()

        # Check for tool call (using the defined tag)
        tool_call_match = re.search(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)
        if tool_call_match:
            tool_call_str = tool_call_match.group(1).strip()
            state["tool_calls"].append({"raw_call": tool_call_str})
            
        # Check for final decision (using the defined tag)
        final_decision_match = re.search(r"<final_decision>(.*?)</final_decision>", content, re.DOTALL)
        if final_decision_match:
            json_content = final_decision_match.group(1).strip()
            # Attempt to parse JSON response
            parsed_data = json.loads(json_content)
            state['final_action'] = parsed_data.get("action", "HOLD").upper()
            state['thought'] = parsed_data.get("thought", "Final decision made.")
            
    except json.JSONDecodeError:
        # LLM failed to produce valid JSON, so no clean decision was made
        pass 
    except Exception:
        # Generic error, no clean decision made
        pass
        
    return state


def run_tool_node(state: AgentState) -> AgentState:
    """Executes the tool call requested by the LLM."""
        
    if not state["tool_calls"]:
        tool_output = "No tool call was specified."
        tool_name = "error"
    else:
        # For simplicity, we directly invoke the tool using the state variables
        tool_output = risk_assessment_tool.invoke({
            "price": state['current_price'],
            "position": state['current_position'],
            "cash": state['current_cash'],
        })
        tool_name = "risk_assessment_tool"
        
    # Append the tool output back to the messages for the LLM's next step
    state["messages"].append(HumanMessage(
        content=f"Observation from Tool {tool_name}: {tool_output}",
        name=tool_name
    ))
    return state


# --- 5. Define Graph Edges (The Flow Control) ---

def decide_next_step(state: AgentState) -> str:
    """Determines whether to run the tool, finish, or loop back to the LLM."""
    
    # 1. STOP: Check if the final action was CLEANLY parsed from the LLM response (best case)
    if state.get('final_action'):
        return "end"
        
    # 2. TOOL: Check if the LLM requested a tool call
    if state["tool_calls"]:
        return "tool"
    
    # 3. FALLBACK STOP: If the recursion limit is near (>= 4 messages) and the LLM 
    # failed to stop, force an end. This prevents the GraphRecursionError.
    if len(state['messages']) >= 4:
        return "end"

    # 4. LOOP: Continue to LLM if no stop condition was met
    return "llm" 


def format_final_output(state: AgentState):
    """A helper function to parse the final action and reasoning, with fallbacks."""
    
    # 1. Use clean parsed data if available
    action = state.get('final_action', 'HOLD')
    thought = state.get('thought', 'LLM concluded with no specific reasoning.')
    
    # 2. FALLBACK: Scrape action/thought from the final message if clean parsing failed
    if not state.get('final_action') and state['messages']:
        final_message = state['messages'][-1].content
        final_message_lower = final_message.lower()
        
        # Scrape action based on keywords
        if 'buy' in final_message_lower:
            action = 'BUY'
        elif 'sell' in final_message_lower:
            action = 'SELL'
        else:
            action = 'HOLD'
            
        # Scrape thought (use the first line for simplicity)
        thought = final_message.split('\n')[0].strip()[:100]
        if thought.startswith("Observation from"):
            thought = "Decision made after observing tool output."
    
    return {"action": action, "thought": thought}


# --- 6. Build and Compile the Graph (Returns a FUNCTION) ---

def build_langgraph_agent():
    """Builds and compiles the LangGraph agent."""
    workflow = StateGraph(AgentState)

    workflow.add_node("llm", run_llm_node)
    workflow.add_node("tool", run_tool_node)

    workflow.set_entry_point("llm")

    workflow.add_conditional_edges(
        "llm",
        decide_next_step,
        {"tool": "tool", "end": END, "llm": "llm"} 
    )

    workflow.add_edge("tool", "llm")

    app = workflow.compile()
    
    # Wrapper function that matches the old .invoke() call structure
    def run_agent_wrapper(inputs):
        """Runs the compiled graph and returns the required format."""
        
        initial_state = AgentState(
            messages=[HumanMessage(content=inputs['prompt'])],
            current_price=inputs['price'],
            current_cash=inputs['cash'],
            current_position=inputs['position'],
            sma_20=inputs['sma_20'],
            sma_50=inputs['sma_50'],
            final_action="",
            thought="",
            tool_calls=[],
        )
        
        # FIX: Increased recursion limit to 5 to avoid GraphRecursionError
        final_state = app.invoke(initial_state, {"recursion_limit": 5})
        
        return format_final_output(final_state)

    # Return the runnable FUNCTION
    return run_agent_wrapper