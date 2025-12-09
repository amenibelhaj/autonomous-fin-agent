# app.py
# FIXES INCLUDED:
# 1. Correct LangGraph function initialization and call.
# 2. Correct Streamlit button and session state management to prevent rerun loops.
# 3. CRITICAL FIX: Changed @st.cache_data to @st.cache_resource on run_backtest_simulation
#    to allow caching of custom Python objects (Agents).

from langgraph_agent import build_langgraph_agent, OLLAMA_STATUS, ollama_client
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from yahooquery import Ticker

st.set_page_config(layout="wide", page_title="Autonomous Trading Agent Dashboard")

# -------------------------
# Build LLM Agent (CACHED for performance)
# -------------------------
@st.cache_resource
def get_runnable_agent():
    # This returns the LangGraph executable FUNCTION
    return build_langgraph_agent()

# The runnable function instance
lg_agent = get_runnable_agent() 

# -------------------------
# Check LLM Status for User Feedback
# -------------------------
try:
    if OLLAMA_STATUS == "Connected" and ollama_client:
        models_list = ollama_client.list()['models']
        # Safely get model name
        model_info = models_list[0]
        model_name = model_info.get('name') or model_info.get('model') or 'phi3:mini'
        
        st.sidebar.success(f"LLM Agent Status: Connected to **{model_name}**")
    else:
        st.sidebar.error(f"LLM Agent Status: {OLLAMA_STATUS}")
except Exception as e:
    # Use a simplified error message here
    st.sidebar.error(f"LLM Client Error: Connection failed. Is Ollama running?")

# -------------------------
# Fetch Yahoo Finance Data (CACHED)
# -------------------------
@st.cache_data
def fetch_yahoo_history(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    try:
        ticker = Ticker(symbol)
        raw = ticker.history(period=period, interval=interval)
        if raw is None or getattr(raw, "empty", True):
             return pd.DataFrame()
        
        if isinstance(raw.index, pd.MultiIndex):
            raw = raw.reset_index()
        date_col = "date" if "date" in raw.columns else [c for c in raw.columns if "date" in c.lower()][0]
        close_col = [c for c in raw.columns if "close" in c.lower()][0]
        df = raw[[date_col, close_col]].copy()
        df.columns = ["date", "Close"]
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
        df.set_index("date", inplace=True)
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.sort_index()
        return df
    except Exception as e:
        st.error(f"Data Fetch Error: Check connectivity or ticker symbol. Details: {e}")
        return pd.DataFrame()

# -------------------------
# Risk Assessment Helper
# -------------------------
def simple_risk_assessment(portfolio_value, cash, position, price):
    exposure = position * price
    total = max(portfolio_value, 1.0)
    ratio = exposure / total
    if ratio > 0.7:
        return ratio, "High"
    elif ratio > 0.4:
        return ratio, "Medium"
    else:
        return ratio, "Low"

# -------------------------
# Base Agent Class
# -------------------------
class BaseAgent:
    def __init__(self, name: str, initial_cash: float):
        self.name = name
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.position = 0
        self.portfolio_history = []
        self.action_history = []
        self.explanation_history = []
        self.risk_history = []

    def observe(self, row: pd.Series):
        self.current_price = float(row["Close"])
        self.row = row

    def reason(self):
        self.next_action = "HOLD"
        self.explanation = "No rule"

    def act(self):
        if self.next_action == "BUY" and self.cash >= self.current_price:
            shares = int(self.cash // self.current_price)
            if shares > 0:
                self.position += shares
                self.cash -= shares * self.current_price
                self.action_history.append(f"BUY {shares} @ {self.current_price:.2f}")
            else:
                self.action_history.append("HOLD")
        elif self.next_action == "SELL" and self.position > 0:
            self.cash += self.position * self.current_price
            self.action_history.append(f"SELL {self.position} @ {self.current_price:.2f}")
            self.position = 0
        else:
            self.action_history.append("HOLD")

        portfolio_value = self.cash + self.position * self.current_price
        ratio, risk = simple_risk_assessment(portfolio_value, self.cash, self.position, self.current_price)
        self.portfolio_history.append(portfolio_value)
        self.risk_history.append(risk)
        self.explanation_history.append(f"{self.next_action} | {self.explanation} | exposure={ratio:.2f} risk={risk}")

# -------------------------
# SMA Agent (LLM-powered)
# -------------------------
class SMAAgent(BaseAgent):
    def __init__(self, name, initial_cash, short=20, long=50):
        super().__init__(name, initial_cash)
        self.short = short
        self.long = long

    def reason(self):
        row = self.row
        s, l = f"SMA_{self.short}", f"SMA_{self.long}"
        
        if pd.isna(row.get(s)) or pd.isna(row.get(l)):
            self.next_action = "HOLD"
            self.explanation = "Not enough SMA data"
            return

        # Optimization: Only call the slow LLM for complex decisions (SMA cross or close)
        sma_diff = abs(row[s] - row[l])
        
        # Define a threshold (e.g., 0.5% difference in price)
        if sma_diff > (0.005 * row["Close"]):
             self.next_action = "HOLD"
             self.explanation = "Market trending strongly; complex analysis bypassed for speed."
             return

        # Call the LangGraph agent wrapper function (lg_agent is the function)
        user_prompt = (
            f"Decide if I should BUY, SELL, or HOLD. "
            f"Explain your final decision in one sentence, referencing the SMAs."
        )

        ai_result = lg_agent({
            "prompt": user_prompt, 
            "price": row["Close"],
            "sma_20": row[s],
            "sma_50": row[l],
            "cash": self.cash,
            "position": self.position,
        })
        
        self.next_action = ai_result.get("action", "HOLD").upper()
        self.explanation = ai_result.get("thought", "LLM reasoning failed or was empty.")

# -------------------------
# Conservative Agent (Benchmark)
# -------------------------
class ConservativeAgent(BaseAgent):
    def reason(self):
        row = self.row
        if pd.isna(row.get("SMA_20")) or pd.isna(row.get("SMA_50")):
            self.next_action = "HOLD"
            self.explanation = "Not enough SMA data"
            return
        
        # Simple crossover strategy with a safety check
        if row["SMA_20"] > row["SMA_50"] and row["Close"] > row.get("PrevClose", row["Close"] - 1):
            self.next_action = "BUY"
            self.explanation = "SMA_20 crossed above SMA_50 and price is rising."
        elif row["SMA_20"] < row["SMA_50"]:
            self.next_action = "SELL"
            self.explanation = "SMA_20 crossed below SMA_50."
        else:
             self.next_action = "HOLD"
             self.explanation = "No clear signal."


# -------------------------
# Metrics and Backtesting Logic
# -------------------------

def compute_metrics(series: pd.Series):
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) < 2:
        return {"total_return_pct":0.0,"cagr_pct":0.0,"max_drawdown_pct":0.0,"sharpe":0.0}
    total_return = series.iloc[-1]/series.iloc[0]-1
    days = (series.index[-1]-series.index[0]).days or 1
    cagr = (1+total_return)**(365.0/days)-1
    daily_ret = series.pct_change().dropna()
    sharpe = (daily_ret.mean()/(daily_ret.std()+1e-9))*np.sqrt(252)
    dd = series/series.cummax()-1
    return {"total_return_pct":total_return*100,"cagr_pct":cagr*100,"max_drawdown_pct":dd.min()*100,"sharpe":sharpe}

# CRITICAL FIX: Changed to @st.cache_resource to allow caching of Agent class instances
@st.cache_resource(show_spinner=False)
def run_backtest_simulation(symbol, period, initial_cash):
    """Runs the full backtest simulation and returns all results."""
    df = fetch_yahoo_history(symbol, period=period)
    
    if df.empty:
        return None

    # Prepare data with indicators
    df["PrevClose"] = df["Close"].shift(1)
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()

    # Initialize agents
    agents = [SMAAgent("LLM Agent", initial_cash), ConservativeAgent("Conservative Agent", initial_cash)]

    # The backtesting loop
    for _, row in df.iterrows():
        for agent in agents:
            agent.observe(row)
            agent.reason()
            agent.act()

    # Consolidate results
    portfolio_df = pd.DataFrame({a.name: a.portfolio_history for a in agents}, index=df.index)
    
    return {
        "portfolio_df": portfolio_df,
        "df": df,
        "agents": agents,
        "initial_cash": initial_cash
    }

# -------------------------
# Streamlit UI Setup
# -------------------------
st.sidebar.header("Agent Controls")
symbol = st.sidebar.text_input("Ticker", "MSFT").upper().strip()
period = st.sidebar.selectbox("Period", ["1y","6mo","3mo","1mo"], index=0)
initial_cash = float(st.sidebar.number_input("Initial cash (USD)", 100000))

# Initialize session state for results
if 'run_finished' not in st.session_state:
    st.session_state.run_finished = False
if 'agent_results' not in st.session_state:
    st.session_state.agent_results = None

button_key = "RunAgentsButton"
run_button = st.sidebar.button("Run Agents", key=button_key)

st.title("Autonomous Trading Agent — Multi-Agent Dashboard")
st.markdown("LLM-powered SMA agent + Conservative agent with risk visualization, actions, and metrics.")

# -------------------------
# Execution Block (Triggered by button)
# -------------------------
if run_button:
    # Clear previous results immediately
    st.session_state.run_finished = False
    st.session_state.agent_results = None
    
    # Run the backtest simulation (cached function)
    with st.spinner(f"Running LLM agent on historical data (this may take a minute for the first run)..."):
        # The function is now cached with @st.cache_resource
        results = run_backtest_simulation(symbol, period, initial_cash)

    if results is None:
        st.error(f"No data available for {symbol} in the selected period.")
        st.session_state.run_finished = True
    else:
        # Store results in session state
        st.session_state.agent_results = results
        st.session_state.run_finished = True
        
        # CRITICAL FIX: Delete the button's key from the session state 
        # to prevent the Streamlit rerun loop.
        del st.session_state[button_key]

        st.rerun()

# -------------------------
# Display Results (Runs only when 'run_finished' is True)
# -------------------------
if st.session_state.run_finished and st.session_state.agent_results is not None:
    results = st.session_state.agent_results
    portfolio_df = results["portfolio_df"]
    df = results["df"]
    agents = results["agents"]
    initial_cash = results["initial_cash"]
    
    # Calculate scaled price for comparison (Buy and Hold benchmark)
    scaled_price = df["Close"]*(initial_cash/df["Close"].iloc[0])
    portfolio_df["ScaledPrice"] = scaled_price

    # Metrics cards
    st.subheader("Performance Metrics")
    cols = st.columns(len(agents) + 1) # +1 for Buy & Hold
    
    # Metrics for Agents
    for i, agent in enumerate(agents):
        m = compute_metrics(portfolio_df[agent.name])
        cagr_color = "green" if m["cagr_pct"]>0 else "red"
        sharpe_color = "green" if m["sharpe"]>0 else "red"
        cols[i].metric(agent.name, f"${agent.portfolio_history[-1]:,.2f}")
        cols[i].markdown(f"Return: **{m['total_return_pct']:.2f}%**")
        cols[i].markdown(f"CAGR: <span style='color:{cagr_color}'>{m['cagr_pct']:.2f}%</span>", unsafe_allow_html=True)
        cols[i].markdown(f"Sharpe: <span style='color:{sharpe_color}'>{m['sharpe']:.2f}</span>", unsafe_allow_html=True)

    # Metrics for Buy and Hold
    m_hold = compute_metrics(scaled_price)
    cols[-1].metric("Buy & Hold", f"${scaled_price.iloc[-1]:,.2f}")
    cols[-1].markdown(f"Return: **{m_hold['total_return_pct']:.2f}%**")
    cols[-1].markdown(f"CAGR: {m_hold['cagr_pct']:.2f}%")
    cols[-1].markdown(f"Sharpe: {m_hold['sharpe']:.2f}")


    # Portfolio Plot
    st.subheader("Portfolio Value Comparison")
    fig = go.Figure()
    for name in portfolio_df.columns:
        if name == "ScaledPrice":
             fig.add_trace(go.Scatter(x=portfolio_df.index, y=portfolio_df[name], mode="lines", name="Buy & Hold (Benchmark)", line=dict(color="black", dash="dash")))
        else:
             fig.add_trace(go.Scatter(x=portfolio_df.index, y=portfolio_df[name], mode="lines", name=name))
             
    fig.update_layout(xaxis_title="Date", yaxis_title="Portfolio Value", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Last 10 Actions Table
    st.subheader("Last 10 Actions & Agent Reasoning")
    for agent in agents:
        st.write(f"### {agent.name}")
        last_idx = min(10, len(agent.action_history))
        df_actions = pd.DataFrame({
            "Date": df.index[-last_idx:],
            "Action": agent.action_history[-last_idx:],
            "Explanation (Thought)": agent.explanation_history[-last_idx:],
            "Risk": agent.risk_history[-last_idx:],
            "Portfolio": [f"${p:,.2f}" for p in agent.portfolio_history[-last_idx:]]
        }).set_index("Date")
        
        def color_risk(r):
            if r=="High": return "background-color: #f28b82;"
            elif r=="Medium": return "background-color: #fbbc04;"
            else: return "background-color: #ccff90;"
            
        st.dataframe(df_actions.style.applymap(color_risk, subset=["Risk"]))

    # CSV download
    combined = portfolio_df.copy()
    combined.index = combined.index.astype(str)
    st.download_button("Download Portfolio Data (CSV)", combined.to_csv(index=True), file_name=f"{symbol}_multi_portfolio.csv")