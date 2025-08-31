# Investment Advisor

## Disclaimer: This project is intended solely for learning and researching AI technology and should not be used as a basis for actual investment decisions. The author assumes no responsibility for any losses incurred if you make investments based on this project. Investment carries risk, please exercise caution.

End-to-end, auditable research-to-execution pipeline for a value-investing, risk-first fund workflow. It produces:

- Research memo, valuation highlights, and scenarios (bear/base/bull)
- Portfolio proposal (target weight, build schedule, liquidity, constraints)
- Risk & compliance checks (VaR, stress, soft caps)
- Trading plan (algo, cost estimate, TCA baseline)
- Optional Markdown report for IC/IR

# Example

[example](examples/en/TESLA/report_2025-08-31.md)

## Architecture Overview

- Orchestrator: drives the pipeline and writes artifacts to `knowledge/`
- Research Agent: generates structured ResearchMemo (thesis, moat, valuation, risks)
- PM Agent: proposes Proposal (target_weight, build_schedule, liquidity, constraints)
- Risk/Compliance: validates VaR/limits and restricted lists (MVP)
- Trading Agent: selects algo and cost estimate
- Report Agent: renders a Markdown report with charts

Artifacts (JSON):

- `knowledge/research/<instrument>/memo_<date>.json`
- `knowledge/proposals/<instrument>/proposal_<date>.json`
- `knowledge/risk/<date>/risk_<instrument>.json`
- `knowledge/compliance/<date>/compliance_<instrument>.json`
- `knowledge/trading/<date>/plan_<instrument>.json`
- Optional: `knowledge/proposals/<instrument>/report_<date>.md` and charts under `figs/`

## Dependencies

- Python 3.11+
- OpenAI API key: `OPENAI_API_KEY` (required)
- Optional MCP servers for market data/search:
  - Financial datasets MCP: `FINANCIAL_DATASETS_API_KEY`
  - Google Search MCP: `GOOGLE_API_KEY`, `GOOGLE_SEARCH_ENGINE_ID`

You can also configure model via `OPENAI_MODEL` (default: gpt-4o-mini).

## Quickstart

1. Create a virtual environment and install packages

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

1. Set environment variables

```bash
export OPENAI_API_KEY=sk-...
# Optional model override
export OPENAI_MODEL=gpt-4o-mini
# Optional MCP keys (if you will use --use-mcp)
export FINANCIAL_DATASETS_API_KEY=...
export GOOGLE_API_KEY=...
export GOOGLE_SEARCH_ENGINE_ID=...
```

1. Run the pipeline (LLM-only path)

```bash
python main.py --instrument "Analyze TESLA" --advice --report --lang en
```

Outputs will be written to `knowledge/`.

## MCP Integration

You can bootstrap them via the init script below, which will clone MCP servers and create `mcp/mcp.json` with local paths.

Run with MCP enabled:

```bash
python main.py --instrument "Analyze TESLA" --use-mcp --advice --report --lang en
```

## Initialize Project (MCP bootstrap)

A convenience script is provided to set up a venv, install Python deps, and (optionally) clone MCP servers.

```bash
bash scripts/init_project.sh
```

The script:

- Creates/activates `venv` and installs `requirements.txt`
- Clones MCP servers into `mcp/vendors/` if URLs are provided
- Builds the Google Search MCP (npm) when present
- Generates `mcp/mcp.json` pointing to the local clones

If you already maintain MCP servers elsewhere, you can keep `mcp/mcp.json` as-is.

## Configuration (Policy)

Baseline risk limits are configurable via environment variables:

- `POLICY_MAX_SINGLE_NAME` (default 0.06)
- `POLICY_SECTOR_CAP` (default 0.20)
- `POLICY_VAR_SOFT_CAP` (default 0.035)

The PM agent enforces these when forming proposals and aligns the build schedule to bridge the gap from existing position -> target.

## Code Layout

- `ai_advisor/schemas.py`: Pydantic schemas for all artifacts
- `ai_advisor/agents.py`: research/PM/trading agents and proposal normalization
- `ai_advisor/risk_compliance.py`: VaR/limits checks (MVP)
- `ai_advisor/orchestrator.py`: pipeline and artifact writing
- `ai_advisor/report_agent.py`: report rendering and chart generation
- `ai_advisor/policy.py`: policy baselines loaded from env
- `main.py`: CLI entry

## Notes

- This repo writes local files only. No trading/execution.
- You can replace models, prompts, risk policy, or add MCP tools with minimal changes to the pipeline.
