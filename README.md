## ProFinView Multi-Agent System

ProFinView is a three-tier multi-agent architecture for financial analysis over S&P 500 data.

- **Tier 1**: Data Ingestion & Feature Engineering (Market Data, Feature, Sentiment)
- **Tier 2**: Analysis & Insight Generation (Analysis, Signal, Position)
- **Tier 3**: Strategy & Execution (Strategy, Execution, Risk)

### Quick Start
- Create and activate a Python 3.10+ environment.
- Install dependencies:
```bash
pip install -r requirements.txt
```
- Optional: set API key for LLM sentiment:
```bash
export XHUB_API_KEY="YOUR_API_KEY"
```
- Run a sample workflow on the S&P 500 JSON folder:
```bash
python -m profinview.cli run technical --data-dir "/Users/alankou/Desktop/ICLR project/sp500_data" --symbols AAPL,MSFT --lookback 180
```

### Project Structure
```
profinview/
  core/            # messaging, schemas, coordinator
  data/            # loaders and normalization
  agents/          # tiered agents
  workflows/       # runnable workflows
  cli.py           # Typer CLI entrypoint
```

### Notes
- Sentiment agent uses env `XHUB_API_KEY` to call `https://api3.xhub.chat/v1/chat/completions`.
- Data loader expects per-symbol JSON or a folder with daily time series including at least date and close.
