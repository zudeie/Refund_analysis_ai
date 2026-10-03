# Vireo Audio – Refund Analysis AI Agent

This is a conversational AI agent I built for **Vireo Audio**. It lets the finance and operations team ask natural language questions about refund data and instantly get exact numbers + interactive charts.

## What it does

The agent has two main abilities:

1. Answers questions about refunds (totals, rankings, comparisons, breakdowns by reason or agent, quarter-over-quarter analysis, etc.) using structured tools.
2. Generates interactive Plotly charts when the user asks for a visualization.

It uses a hybrid approach:
- Structured tools for reliable numbers
- A dynamic visualization tool that writes pandas + Plotly code only when a chart is requested

The agent is designed to never invent numbers, always calls a tool.

## Tech Stack

- Streamlit for the chat UI
- LangGraph for the agent orchestration
- LangChain tools
- Local Ollama model (Qwen3.5-9B) as the LLM (can be switched to Groq)
- Pandas for data processing
- Plotly for interactive charts
- Python-dotenv for environment variables

## Project Structure
```
Refund_analysis_ai/
├── app.py                      # Streamlit chat UI
├── Data_cleaning_pipeline.py   # One-time data cleaning script
├── agent/
│   ├── agent.py                # LangGraph agent definition
│   ├── tools.py                # Structured tools + visualization tool
│   └── prompts.py              # System prompt
├── cleaned_data/               # Cleaned CSVs used by the agent
│   ├── clean_refunds.csv       # Main file
│   ├── agents.csv
│   ├── orders.csv
│   ├── customers.csv
│   └── products.csv
├── data/                       # Original raw CSVs
├── requirements.txt
└── .env                        # API keys 
```

## How to run
1. Clone the repo

```bash
git clone https://github.com/zudeie/Refund_analysis_ai.git
cd Refund_analysis_ai
```
2. Create a virtual environment and install dependencies

```
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```
3. Create a .env file with these variables: 

```
LlmProvider_API_KEY=your_API_key_here
```
Note: The current code uses a local Ollama model.

If you want to use desired LLM provider, open agent/agent.py and change the LLM initialization.

4. Clean the data (run once):
```
python Data_cleaning_pipeline.py
```
This will:

* Remove duplicate tickets
* Convert legacy_fd amounts from paise → rupees
* Keep only real refunds
* Add helper columns (year_month, quarter, etc.)
* Save everything into the cleaned_data/ folder

5. Start the app :
``` 
streamlit run app.py
```
Open the URL that appears (usually http://localhost:8501).

## Example Questions,

You can use the sidebar buttons or type these yourself:

* What is the total refund amount for 2025Q1?
* Show me the top 5 agents who gave the most refunds
* Compare 2025Q1 vs 2025Q2
* Give me refund breakdown by reason code for 2025Q4
* Show a bar chart of refunds by reason for 2025Q3
* Visualize top 10 agents by refund amount in 2026Q1
* Which agent issued the most refunds overall?
* Give me monthly refunds by agent for 2025-03

## Notes

* The LLM is currently set to a local Ollama model (Qwen3.5-9B). Stronger models give more consistent tool calling.
* If the Streamlit UI starts throwing errors about duplicate charts or string indices, just click Clear Conversation in the sidebar.
* Chat history lives only in the current Streamlit session (it is lost when you refresh the page).
* All money values are shown in Indian Rupees (₹).

## Future improvements I want to add

* Switch to a stronger cloud model by default
* Add simple authentication
* Persistent conversation history
* Automated evaluation suite
* Multi agent version (one agent for numbers, one for charts) if question complexity grows