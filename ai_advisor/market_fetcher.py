from __future__ import annotations

import asyncio
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dateutil.relativedelta import relativedelta
from .mcp_client import get_mcp_client, wrap_mcp_tools_for_logging, MCPNotAvailable
from .storage import ensure_dir


async def fetch_and_write_market_artifacts(
    instrument: str,
    ticker:str,
    as_of: str,
    root: str = "knowledge",
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Optional[str]]:
    """Assemble market artifacts from existing knowledge outputs; no new fetching.

    Preference order:
    1) knowledge/market/{instrument}/index_{as_of}.json if present
    2) Synthesize minimal artifacts from proposals/{instrument}/advice_*.json and report_*.json
    """

    out: Dict[str, Optional[str]] = {
        "quote": None,
        "prices": None,
        "financials_income": None,
        "news": None,
        "filings": None,
        "insiders": None,
        "ownership": None,
    }

    log = logger or logging.getLogger("market.fetch")
    base = Path(root) / "market" / instrument

    # Prefer pre-built index
    index_path = base / f"index_{as_of}.json"
    index: Optional[Dict[str, Any]] = None
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            index = None

    # Fallback: synthesize from advice/report
    if index is None:
        advice_path = Path(root) / "proposals" / instrument / f"advice_{as_of}.json"
        report_path = Path(root) / "proposals" / instrument / f"report_{as_of}.json"
        index = {
            "quote": None,
            "prices": None,
            "financials_income": None,
            "news": None,
            "filings": None,
            "insiders": None,
            "ownership": None,
        }
        # News from advice headlines
        try:
            if advice_path.exists():
                advice = json.loads(advice_path.read_text(encoding="utf-8"))
                headlines = advice.get("market_context", {}).get("sentiment", {}).get("headlines")
                if isinstance(headlines, list) and headlines:
                    index["news"] = {
                        "headlines": headlines,
                        "source": "advice.market_context.sentiment.headlines",
                    }
        except Exception:
            pass
        # Financial snippets from report
        try:
            if report_path.exists():
                report = json.loads(report_path.read_text(encoding="utf-8"))
                memo = report.get("researchMemo", {})
                ue = memo.get("unit_economics")
                valuation = memo.get("valuation")
                fin: Dict[str, Any] = {}
                if isinstance(ue, dict):
                    fin["unit_economics"] = ue
                if isinstance(valuation, dict):
                    fin["valuation"] = valuation
                if fin:
                    index["financials_income"] = fin
        except Exception:
            pass

    # Fetch via MCP tools and write artifacts
    async def _get_tools(selected: Optional[List[str]] = None, log: Optional[logging.Logger] = None):
        client = get_mcp_client()
        tools = await client.get_tools()
        if selected:
            tools = [t for t in tools if getattr(t, "name", "") in set(selected)]
        if log:
            tools = wrap_mcp_tools_for_logging(tools, log)
        return {getattr(t, "name", f"tool_{i}"): t for i, t in enumerate(tools)}

    async def _safe_call(tool, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            if hasattr(tool, "ainvoke"):
                res = await tool.ainvoke(payload)
            else:
                res = tool.invoke(payload)
            if isinstance(res, dict):
                return res
            if isinstance(res, str):
                try:
                    return json.loads(res)
                except Exception:
                    return {"raw": res}
            return {"raw": res}
        except Exception as e:
            print(e)
            return None

    try:
        tools = await _get_tools(
            selected=[
                "get_current_stock_price",
                "get_historical_stock_prices",
                "get_income_statements",
                "get_company_news",
                "get_sec_filings",
                "get_balance_sheets",
                "get_cash_flow_statements",
            ],
            log=log,
        )
    except MCPNotAvailable:
        return out

    ensure_dir(base)

    async def _gather():
        return await asyncio.gather(
            _safe_call(tools.get("get_current_stock_price"), {"ticker": ticker}),
            _safe_call(tools.get("get_historical_stock_prices"), {"ticker": ticker, "start_date": (datetime.now() - relativedelta(days=120)).strftime('%Y-%m-%d'), "end_date": datetime.today().strftime('%Y-%m-%d'), "interval": "day"}),
            _safe_call(tools.get("get_historical_stock_prices"), {"ticker": ticker, "start_date": (datetime.now() - relativedelta(years=20)).strftime('%Y-%m-%d'), "end_date": datetime.today().strftime('%Y-%m-%d'), "interval": "month"}),
            _safe_call(tools.get("get_income_statements"), {"ticker": ticker}),
            _safe_call(tools.get("get_company_news"), {"ticker": ticker}),
            _safe_call(tools.get("get_sec_filings"), {"ticker": ticker}),
        )

    quote, prices, year_prices, income, news, filings = await _gather()

    def _write(name: str, obj: Optional[Dict[str, Any]]) -> Optional[str]:
        if not obj:
            return None
        path = base / f"{name}_{as_of}.json"
        try:
            path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
            return str(path)
        except Exception:
            return None

    out["quote"] = _write("quote", quote)
    out["prices"] = _write("prices", prices)
    out["year_prices"] = _write("year_prices", year_prices)
    out["financials_income"] = _write("financials_income", income)
    out["news"] = _write("news", news)
    out["filings"] = _write("filings", filings)

    # Compute ADV(30/90) if volume is present in historical prices
    adv_30: Optional[float] = None
    adv_90: Optional[float] = None
    try:
        def _extract_vols(obj: Dict[str, Any]) -> List[float]:
            vols: List[float] = []
            if isinstance(obj, dict):
                raw = obj.get("raw")
                if isinstance(raw, list):
                    for blk in raw:
                        data = blk.get("data") if isinstance(blk, dict) else None
                        if isinstance(data, list):
                            for row in data:
                                v = row.get("volume") or row.get("v")
                                if isinstance(v, (int, float)):
                                    vols.append(float(v))
                data = obj.get("data")
                if isinstance(data, list):
                    for row in data:
                        v = row.get("volume") or row.get("v")
                        if isinstance(v, (int, float)):
                            vols.append(float(v))
            return vols
        vols = _extract_vols(prices or {}) if isinstance(prices, dict) else []
        if vols:
            if len(vols) >= 30:
                adv_30 = sum(vols[-30:]) / 30.0
            if len(vols) >= 90:
                adv_90 = sum(vols[-90:]) / 90.0
    except Exception:
        adv_30 = adv_90 = None

    # Write/merge index
    try:
        idx: Dict[str, Any] = index or {}
        if adv_30 is not None:
            idx["adv_30"] = adv_30
        if adv_90 is not None:
            idx["adv_90"] = adv_90
        if any(v is not None for v in (out.get("quote"), out.get("prices"), out.get("financials_income"), out.get("news"), out.get("filings"))):
            idx["artifacts"] = {k: v for k, v in out.items() if v}
        ensure_dir(base)
        index_path.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    return out


__all__ = ["fetch_and_write_market_artifacts"]
