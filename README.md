# ULLM-Project
Diagnostic Score for Validating the Huggingface Data Analyst Agent Maria Dirnberger (6638410), Sarthak Bhardwaj (6648707), Sumitrra Bala (6642271)
<div align="center">
  <img src="./jupyter-agent-2.png" alt="Jupyter Agent Logo" width="240" />
  <h1>Jupyter Agent 2</h1>
  <p>
    An autonomous, streaming Jupyter notebook generation & execution agent powered by<br>
    Hugging Face Inference (chat completions), an E2B sandbox for code execution,<br>
    and a lightweight heuristic evaluation toolkit.
  </p>
  <p>
    <a href="https://huggingface.co/spaces/lvwerra/jupyter-agent-2">HF Space</a> ·
    <a href="#quick-start">Quick Start</a> ·
    <a href="#architecture">Architecture</a> ·
    <a href="#evaluation">Evaluation</a> ·
    <a href="#contributing">Contributing</a>
  </p>
</div>

---

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [Requirements](#requirements)
6. [Quick Start](#quick-start)
7. [Configuration](#configuration)
8. [How It Works (Agent Loop)](#how-it-works-agent-loop)
9. [Evaluation](#evaluation)
10. [Extending the Agent](#extending-the-agent)
11. [Troubleshooting](#troubleshooting)
12. [Roadmap](#roadmap)
13. [Contributing](#contributing)
14. [License](#license)

## Overview
Jupyter Agent 2 is an experimental autonomous coding assistant that:

* Streams a live, styled Jupyter notebook to the browser (via Gradio Blocks)
* Calls an LLM (default: `Qwen3-Coder-480B-A35B-Instruct` through Hugging Face's router) using the Chat Completions API with tool/function calls
* Executes generated Python code incrementally inside an **E2B** secure sandbox
* Persists state between code cells (variables, imports, outputs)
* Renders a rich conversational + execution trace (system / user / assistant / tool outputs)
* Automatically appends a lightweight **heuristic evaluation** summary of the final notebook (EDA, preparation, modelling, evaluation)

This repository also includes utility notebooks and an evaluation script (`eval_tool.py`) usable standalone to score arbitrary notebooks + message histories.

## Key Features
* Incremental code generation & execution loop with streaming UI
* Built-in heuristic scoring: Data Understanding, Data Preparation, Modelling, Evaluation
* Tool calling via function schema (currently: execute code cell; easy to add more)
* Cleans oversized message payloads to avoid API size issues (`clean_messages_for_api`)
* Sandbox timeout visualization / countdown in rendered notebook
* Simple dependency footprint (pure Python + standard ML/data stack)
* Extensible patterns (add tools, evaluation heuristics, models)

## Architecture
High-level flow:

1. User supplies an instruction (and optional file uploads)
2. System prompt lists available files & base environment packages
3. Agent loop (`run_interactive_notebook`) calls LLM with conversation + tool schema
4. LLM may return a `tool_call` requesting code execution
5. Code is executed in an **E2B Sandbox** → outputs (stdout, rich display data, errors) captured
6. Notebook builder (`JupyterNotebook`) appends formatted markdown/code/output cells
7. Loop continues until no further tool calls or max turns reached
8. Final notebook HTML is passed to `eval_tool.evaluate_session` → scores appended

Component responsibilities:

| Component | Role |
|-----------|------|
| `app.py` | Gradio UI, session handling, sandbox lifecycle, evaluation wrapper |
| `utils.py` | Agent loop, tool schema, execution parsing, message sanitation |
| `jupyter_handler.py` | Notebook assembly, styling, countdown overlay, HTML export |
| `eval_tool.py` | Heuristic extraction & scoring (regex-based) |
| `tmp/` | Per-session notebook artifacts |
| `history.json` | Example message history for evaluation |

## Project Structure
```
app.py                 # Gradio interface & evaluation integration
utils.py               # Agent loop & tool execution
jupyter_handler.py      # Notebook model & HTML rendering
eval_tool.py           # Heuristic scoring tool (standalone CLI)
requirements.txt       # Python dependencies
history.json           # Sample conversation for scoring
tmp/                   # Session notebook artifacts
*.ipynb                # Demo / reference notebooks
```

## Requirements
* Python 3.11+ (tested with 3.11 / 3.12)
* (Optional) HF token (for router) → set `HF_TOKEN`
* (Optional) E2B API key → set `E2B_API_KEY` (falls back to test key)

Install deps:
```bash
pip install -r requirements.txt
```

## Quick Start
Run locally:
```bash
export HF_TOKEN=hf_...        # or set in PowerShell: $env:HF_TOKEN="hf_xxx"
export E2B_API_KEY=xyz        # optional
python app.py                 # launches Gradio UI
```
Open the printed local URL. Enter an analytical task (e.g. *"Explore the predict-calorie dataset and build a regression model"*). Watch notebook stream in real time.

### Running a Standalone Evaluation
Convert a notebook to HTML (if `.ipynb`):
```bash
python -m nbconvert --to html jupyter-agent_5.ipynb
```
Then score:
```bash
python eval_tool.py --notebook jupyter-agent_5.html --history history.json --out evaluation_report.json
```

## Configuration
Environment variables:

| Name | Purpose | Default |
|------|---------|---------|
| `HF_TOKEN` | Auth for Hugging Face Inference Router | None (unauth may rate-limit) |
| `E2B_API_KEY` | E2B sandbox execution | `e2b_test_key` |
| `MODEL_NAME` | (If you refactor) override model id | Hardcoded in `app.py` |
| `SANDBOX_TIMEOUT` | Execution sandbox TTL (s) | 420 |

You can also modify the system prompt template in `app.py` (`DEFAULT_SYSTEM_PROMPT`).

## How It Works (Agent Loop)
Core loop (`run_interactive_notebook`):

1. Render initial notebook (generating state)
2. Call HF Chat Completions with messages + `TOOLS` schema
3. If tool call returned → parse, add code cell placeholder → execute in sandbox → append output cell
4. Append tool output as a `tool` role message (with raw execution for notebook fidelity)
5. Repeat until assistant returns plain text (no tool calls) or max turns reached
6. Final render (mode=`done`)
7. Wrapper in `app.py` post-processes with `eval_tool` and appends JSON scores block

### Tool Schema
Currently only one tool is exposed:
```json
{
  "name": "add_and_execute_jupyter_code_cell",
  "parameters": {"code": "string"}
}
```
Add more by extending `TOOLS` in `utils.py` (and implementing handling logic in the loop).

## Evaluation
The heuristic scoring (`eval_tool.py`) inspects notebook HTML + message history using regex pattern groups:
* Data Understanding (EDA calls, summary stats, visualizations)
* Data Preparation (imputation, scaling, encoding, split, feature eng, pipeline)
* Modelling (model families, .fit(), CV/search) 
* Evaluation (metrics, diagnostics, interpretability)

Each dimension gets a coverage + depth composite score (0–1). Optional bonuses:
* Intent keywords in message history
* Artifact presence (e.g. saved model, metrics file, plots) if `--save-dir` provided

CLI:
```bash
python eval_tool.py --notebook <path_or_raw_html> --history <path_or_json> --out report.json
```

Example output snippet:
```json
{
  "data_understanding": {"score": 0.42, ...},
  "data_preparation": {"score": 0.55, ...},
  "overall_score": 0.48
}
```

## Extending the Agent
Ideas:
* Add SQL execution tool (DuckDB, SQLite)
* Add data upload profiling (ydata-profiling / pandas-profiling) artifacts
* Implement retry/backoff & partial cell re-execution
* Integrate vector store for retrieval-augmented prompting
* Replace regex heuristics with embedding-based semantic scoring
* Output export formats: PDF, Markdown summary, dashboard generation

