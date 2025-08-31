from __future__ import annotations

from datetime import date, datetime
import logging
from pathlib import Path
from typing import Optional
import json

from .agents import generate_research_memo, plan_trading, propose_portfolio
from .risk_compliance import compute_risk_check, run_compliance_check
# MCP tools are now handled inside agents via a REACT agent with tools; no direct calls here.
from .schemas import Decision, InvestmentCase
from .storage import write_json
# MCP dataset-specific wrappers removed; use generic tools via mcp_client
from .advisor import generate_investment_advice
from .report_agent import generate_report
from .market_fetcher import fetch_and_write_market_artifacts


def _slugify(text: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text).strip("_")[:60]


logger = logging.getLogger("orchestrator")


async def run_pipeline(
    idea: str,
    instrument_id: Optional[str] = None,
    root: str = "knowledge",
    use_mcp: bool = False,
    generate_advice: bool = False,
    generate_report_md: bool = False,
    language: str = "en",
) -> InvestmentCase:
    as_of = date.today().isoformat()
    instrument = instrument_id or _slugify(idea) or "IDEA"
    # Prepare root path early (used by optional MCP writes)
    root_path = Path(root)

    logger.info("[1/8] Generating research memo (lang=%s)", language)
    research = await generate_research_memo(idea, language=language, use_mcp=use_mcp)

    # Optionally enrich with MCP market data (best-effort, non-fatal)
    pm_trading_context = {}
    prices = None
    if use_mcp and research.ticker_symbol:
        logger.info("[2/8] Fetching market artifacts via tools")
        try:
            logger.info(f"Fetching market artifacts for {research.ticker_symbol} as of {as_of}")
            artifacts = await fetch_and_write_market_artifacts(instrument, research.ticker_symbol, as_of, root, logger=logger)
            # Persist a tiny index for audit and convenience
            try:
                idx_path = Path(root) / "market" / instrument / f"index_{as_of}.json"
                write_json(idx_path, artifacts)
            except Exception:
                pass
            # If prices were fetched, load once and pass to risk to avoid re-read later
            try:
                p_path = artifacts.get("prices")
                if p_path:
                    prices = json.loads(Path(p_path).read_text(encoding="utf-8"))
            except Exception:
                prices = None
            pm_trading_context["market_artifacts"] = artifacts
        except Exception as e:
            logger.warning("Market artifact fetch failed: %s", e)
    # Optionally generate investment advice BEFORE proposal, so proposal can consider current position size
    advice_obj = None
    if generate_advice and instrument:
        try:
            logger.info("[3/8] Generating investment advice (lang=%s)", language)
            advice_obj = await generate_investment_advice(instrument, use_mcp=use_mcp, language=language)
            # make available to portfolio/trading steps
            try:
                pm_trading_context["existing_position_size"] = advice_obj.position_size
            except Exception:
                pass
        except Exception:
            advice_obj = None

    logger.info("[4/8] Proposing portfolio (lang=%s)", language)
    proposal = await propose_portfolio(research, context=pm_trading_context or None, language=language, use_mcp=use_mcp)
    logger.info("[5/8] Computing risk & compliance")
    risk = compute_risk_check(research, proposal, instrument=instrument, as_of=as_of, root=root, prices=prices)
    compliance = run_compliance_check(research, proposal)
    logger.info("[6/8] Planning trading")
    trading = await plan_trading(research, proposal, context=pm_trading_context or None, language=language, use_mcp=use_mcp)

    if (not risk.breach) and (not compliance.restricted):
        status = "IC_APPROVED"
    elif compliance.restricted:
        status = "IC_REJECTED"
    else:
        status = "NEEDS_REVISION"

    decision = Decision(status=status, approvers=["CIO", "Risk"], timestamp=datetime.utcnow().isoformat())

    case = InvestmentCase(
        instrumentId=instrument,
        asOf=as_of,
        researchMemo=research,
        proposal=proposal,
        riskCheck=risk,
        complianceCheck=compliance,
        tradingPlan=trading,
        decision=decision,
    )

    logger.info("[7/8] Writing artifacts")
    # Write artifacts following the README structure
    write_json(root_path / "research" / instrument / f"memo_{as_of}.json", research.model_dump())
    # store only the proposal in proposals/ for clean contract; full report is written by main.py
    write_json(root_path / "proposals" / instrument / f"proposal_{as_of}.json", proposal.model_dump())
    write_json(root_path / "risk" / as_of / f"risk_{instrument}.json", risk.model_dump())
    write_json(root_path / "compliance" / as_of / f"compliance_{instrument}.json", compliance.model_dump())
    write_json(root_path / "trading" / as_of / f"plan_{instrument}.json", trading.model_dump())

    # Persist advice if available
    if advice_obj is not None:
        try:
            write_json(root_path / "proposals" / instrument / f"advice_{as_of}.json", advice_obj.model_dump())
        except Exception:
            pass

    if generate_report_md and instrument:
        try:
            logger.info("[8/8] Rendering Markdown report")
            report_md = root_path / "proposals" / instrument / f"report_{as_of}.md"
            generate_report(case, advice_obj, str(report_md), language=language, use_mcp=use_mcp)
        except Exception:
            pass

    return case
