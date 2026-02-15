# Agent core

## Quickstart

### Requirements
- Pythonß
- OpenAI API key (or Ollama for local models)

### Install the app and library
Clone the repo and install in editable mode:
```bash
pip install -e .
```

### Run the AI Pipeline Studio app
```bash
streamlit run apps/ai-pipeline-studio-app/app.py
```

## Library Overview

The repository includes both the **AI Pipeline Studio** app and the underlying **AI Data Science Team** library. The library provides agent building blocks and multi-agent workflows for:
- Data loading and inspection
- Cleaning, wrangling, and feature engineering
- Visualization and EDA
- Modeling and evaluation (H2O + MLflow tools)
- SQL database interaction

### Agents (Snapshot)

Agent examples live in `examples/`. Notable agents:
- Data Loader Tools Agent
- Data Wrangling Agent
- Data Cleaning Agent
- Data Visualization Agent
- EDA Tools Agent
- Feature Engineering Agent
- SQL Database Agent
- H2O ML Agent
- MLflow Tools Agent
- Multi-agent workflows (e.g., Pandas Data Analyst, SQL Data Analyst)
- Supervisor Agent (oversees other agents)
- Custom tools for data science tasks

## Apps

See all apps in `apps/`. Notable apps:
- AI Pipeline Studio: `apps/ai-pipeline-studio-app/`
- EDA Explorer App: `apps/exploratory-copilot-app/`
- Pandas Data Analyst App: `apps/pandas-data-analyst-app/`