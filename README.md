# Autonomous Trading Agent

A multi-agent backtesting dashboard where an **LLM agent built with LangGraph** decides when to BUY, SELL or HOLD a stock, and is compared against a rule-based agent and a buy-and-hold benchmark. The language model runs **fully locally** through Ollama — no API key, no data leaving your machine.

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1.0-1C3C3C)
![Streamlit](https://img.shields.io/badge/Streamlit-1.52-FF4B4B?logo=streamlit&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-phi3:mini-000000)

## How it works

```
Yahoo Finance prices ──► indicators (SMA-20, SMA-50) ──► agents decide each day ──► portfolio & metrics
                                                          │
                          ┌───────────────────────────────┼───────────────────────────┐
                          ▼                               ▼                           ▼
                    LLM Agent (LangGraph)        Conservative Agent            Buy & Hold
                    reason → call tool →         SMA crossover rules           benchmark
                    decide (ReAct loop)
```

**LLM agent (`langgraph_agent.py`)**
- A LangGraph state machine with two nodes: an **LLM node** that reasons about price, moving averages and holdings, and a **tool node** that runs a `risk_assessment_tool` (exposure ratio → Low / Medium / High risk).
- The model answers in a structured `<final_decision>` JSON block; the graph parses it, with fallbacks and a recursion limit so it always terminates.
- To keep backtests fast, the LLM is only called when the two moving averages are close (a potential crossover). In clear trends the agent holds without a model call.

**Dashboard (`app.py`)**
- Pick any ticker and period, set starting cash, and run all agents over the historical data.
- Compares agents on **total return, CAGR, Sharpe ratio and max drawdown**, with an interactive Plotly chart against buy-and-hold.
- Shows each agent's last 10 actions with its **reasoning and risk level**, and exports the results as CSV.

## Tech stack

Python · LangGraph · LangChain · Ollama (`phi3:mini`) · Streamlit · Plotly · pandas · NumPy · yahooquery

## Run it locally

**1. Install and start Ollama**, then pull the model:

```bash
ollama pull phi3:mini
```

**2. Set up the project**

```bash
git clone https://github.com/amenibelhaj/autonomous-fin-agent.git
cd autonomous-fin-agent
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Launch the dashboard**

```bash
streamlit run app.py
```

To use a different local model, set `OLLAMA_MODEL` (for example `OLLAMA_MODEL=llama3.1 streamlit run app.py`).

## Project structure

```
app.py               Streamlit dashboard, agents, backtest loop, metrics
langgraph_agent.py   LangGraph LLM agent and risk-assessment tool
analysis.ipynb       Exploratory analysis of the agents' results
working.ipynb        Data and indicator prototyping
requirements.txt     Python dependencies
```

## Disclaimer

This is an educational project for experimenting with LLM agents. It is not financial advice and should not be used to trade real money.
