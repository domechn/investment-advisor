from __future__ import annotations

from typing import Optional, List, Tuple, Any
from pathlib import Path
import json
import os

from .schemas import InvestmentCase, InvestmentAdvice
from .storage import write_text, ensure_dir
from langchain_openai import ChatOpenAI  # type: ignore
from langchain_core.prompts import ChatPromptTemplate  # type: ignore
# Report agent must not call any MCP tools.


def _load_market_artifacts(root: Path, instrument: str, as_of: str) -> dict:
    out = {}
    base = root / "market" / instrument
    # If no artifacts folder, return empty (all market context is optional for report rendering)
    if not base.exists():
        return out
    def read_json(name: str):
        p = base / name
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None
    out["quote"] = read_json(f"quote_{as_of}.json")
    out["prices"] = read_json(f"prices_{as_of}.json")
    out["financials_income"] = read_json(f"financials_income_{as_of}.json")
    out["news"] = read_json(f"news_{as_of}.json")
    out["filings"] = read_json(f"filings_{as_of}.json")
    out["index"] = read_json(f"index_{as_of}.json")
    # Optional additional datasets (best-effort)
    out["insiders"] = read_json(f"insiders_{as_of}.json")
    out["ownership"] = read_json(f"ownership_{as_of}.json")
    return out


def _extract_series_from_prices(prices: dict | list) -> Tuple[List[float], List[str]]:
    """Extract close prices and dates from various shapes.

    Supported inputs:
    - list of OHLC rows: [{close, date|time|timestamp|time_milliseconds, ...}, ...]
    - dict with {raw: [{data: [rows...]}]}
    - dict with {data: [rows...]}
    - dict with {series: [numbers...]} (fallback, dates as indices)
    """
    closes: List[float] = []
    dates: List[str] = []

    def _push_row(row: dict):
        c = row.get("close") or row.get("price") or row.get("c")
        d = row.get("date") or row.get("time") or row.get("timestamp") or row.get("time_milliseconds")
        if isinstance(c, (int, float)):
            closes.append(float(c))
            # Normalize date field
            if isinstance(d, (int, float)):
                # treat as epoch ms if large
                if d > 10_000_000_000:  # ms
                    try:
                        from datetime import datetime, timezone
                        dates.append(datetime.fromtimestamp(d / 1000.0, tz=timezone.utc).isoformat())
                    except Exception:
                        dates.append(str(int(d)))
                else:
                    try:
                        from datetime import datetime, timezone
                        dates.append(datetime.fromtimestamp(d, tz=timezone.utc).isoformat())
                    except Exception:
                        dates.append(str(d))
            else:
                dates.append(str(d) if d is not None else "")

    # Case 1: list of rows
    if isinstance(prices, list):
        for row in prices:
            if isinstance(row, dict):
                _push_row(row)
        return closes, dates

    # Case 2: dict with raw blocks
    if isinstance(prices, dict):
        raw = prices.get("raw")
        if isinstance(raw, list):
            for blk in raw:
                data = blk.get("data") if isinstance(blk, dict) else None
                if isinstance(data, list):
                    for row in data:
                        if isinstance(row, dict):
                            _push_row(row)
            return closes, dates
        # Case 3: dict with data directly
        data = prices.get("data")
        if isinstance(data, list):
            for row in data:
                if isinstance(row, dict):
                    _push_row(row)
            return closes, dates
        # Case 4: numeric series
        series = prices.get("series")
        if isinstance(series, list):
            for i, v in enumerate(series):
                if isinstance(v, (int, float)):
                    closes.append(float(v))
                    dates.append(str(i))
            return closes, dates

    return closes, dates


# Note: Volumes/ADV should be fetched by the agent via tools; no code-side fetching here.


def _compute_day_change_from_series(closes: List[float]) -> Optional[float]:
    if len(closes) < 2:
        return None
    prev, last = closes[-2], closes[-1]
    if prev > 0:
        return (last / prev) - 1.0
    return None


def _last_nonempty(xs: List[Any]) -> Optional[Any]:
    for v in reversed(xs or []):
        if v is not None and v != "":
            return v
    return None


def _six_month_performance(closes: List[float]) -> Optional[float]:
    if len(closes) < 22:
        return None
    # approx 6M ~ 126 trading days
    window = closes[-126:] if len(closes) >= 126 else closes
    if window and window[0] > 0:
        return (window[-1] / window[0]) - 1.0
    return None


def _summarize_quote(quote: dict) -> str:
    # Avoid conflicting numbers; prefer percentage if present
    ticker = quote.get("ticker")
    t = quote.get("time")
    p = quote.get("price")
    ch_pct = quote.get("day_change_percent")
    if ticker and p is not None:
        if t and isinstance(ch_pct, (int, float)):
            try:
                return f"{ticker} latest price ({t}): {p} ({ch_pct:+.2%})"
            except Exception:
                return f"{ticker} latest price ({t}): {p}"
        if t:
            return f"{ticker} latest price ({t}): {p}"
        return f"{ticker} latest price: {p}"
    return ""

def _summarize_news(news: list | dict | None, topn: int = 5) -> List[str]:
    items = []
    rows: List[dict] = []
    if isinstance(news, list):
        rows = [r for r in news if isinstance(r, dict)]
    elif isinstance(news, dict):
        if isinstance(news.get("raw"), list):
            for blk in news.get("raw"):
                data = blk.get("data") if isinstance(blk, dict) else None
                if isinstance(data, list):
                    rows.extend([r for r in data if isinstance(r, dict)])
        elif isinstance(news.get("data"), list):
            rows = [r for r in news.get("data") if isinstance(r, dict)]
    for blk in rows:
        title = blk.get("title") or ""
        author = blk.get("author") or ""
        source = blk.get("source") or ""
        date = blk.get("date") or ""
        url = blk.get("url") or ""
        sentiment = blk.get("sentiment") or ""

        items.append(f"- [{date}] {title} ({source}) [{author}]({url}) [{sentiment}]")
        if len(items) >= topn:
            return items[:topn]
    return items[:topn]


def _summarize_filings(filings: list | dict | None, topn: int = 5) -> List[str]:
    items: List[str] = []
    rows: List[dict] = []
    if isinstance(filings, list):
        rows = [r for r in filings if isinstance(r, dict)]
    elif isinstance(filings, dict):
        if isinstance(filings.get("raw"), list):
            for blk in filings.get("raw"):
                data = blk.get("data") if isinstance(blk, dict) else None
                if isinstance(data, list):
                    rows.extend([r for r in data if isinstance(r, dict)])
        elif isinstance(filings.get("data"), list):
            rows = [r for r in filings.get("data") if isinstance(r, dict)]
    for filing in rows:
        report_date = filing.get("report_date") or filing.get("date") or ""
        accession_number = filing.get("accession_number") or filing.get("accession") or ""
        filing_type = filing.get("filing_type") or filing.get("type") or ""
        url = filing.get("url") or ""
        if url:
            items.append(f"- {report_date}, Accession: {accession_number}, Filing Type: {filing_type}, Link: {url}")
        else:
            items.append(f"- {report_date}, Accession: {accession_number}, Filing Type: {filing_type}")
        if len(items) >= topn:
            return items[:topn]
    return items[:topn]


def _extract_references(market: dict, rm_sources: List[dict]) -> List[str]:
    """Collect clickable references from research sources, news and filings.
    Returns a list of Markdown bullet items like - [Title](URL).
    """
    refs: List[str] = []
    # From research memo sources
    for s in rm_sources or []:
        try:
            url = s.get("url") or s.get("link")
            accession_number = s.get("title") or s.get("type") or (url or "")
            if url:
                refs.append(f"- [{accession_number}]({url})")
        except Exception:
            continue
    # From news
    news = market.get("news")
    # Support both list-shaped news and dict-shaped with raw/data
    try:
        news_rows: List[dict] = []
        if isinstance(news, list):
            news_rows = [r for r in news if isinstance(r, dict)]
        elif isinstance(news, dict):
            if isinstance(news.get("raw"), list):
                for blk in news.get("raw"):
                    data = blk.get("data") if isinstance(blk, dict) else None
                    if isinstance(data, list):
                        news_rows.extend([r for r in data if isinstance(r, dict)])
            elif isinstance(news.get("data"), list):
                news_rows = [r for r in news.get("data") if isinstance(r, dict)]
        for row in news_rows[:20]:
            url = row.get("url")
            accession_number = row.get("title")
            if url:
                refs.append(f"- [{accession_number}]({url})")
    except Exception:
        pass
    # From filings
    filings = market.get("filings")
    try:
        frows: List[dict] = []
        if isinstance(filings, list):
            frows = [r for r in filings if isinstance(r, dict)]
        elif isinstance(filings, dict):
            if isinstance(filings.get("raw"), list):
                for blk in filings.get("raw"):
                    data = blk.get("data") if isinstance(blk, dict) else None
                    if isinstance(data, list):
                        frows.extend([r for r in data if isinstance(r, dict)])
            elif isinstance(filings.get("data"), list):
                frows = [r for r in filings.get("data") if isinstance(r, dict)]
        for row in frows[:20]:
            url = row.get("url")
            accession_number = row.get("accession_number") or row.get("title")
            if url:
                refs.append(f"- [{accession_number}]({url})")
    except Exception:
        pass
    # De-duplicate while preserving order
    seen = set()
    unique: List[str] = []
    for r in refs:
        if r not in seen:
            unique.append(r)
            seen.add(r)
    return unique[:50]


def _compute_returns(closes: List[float]) -> List[float]:
    rets: List[float] = []
    for i in range(1, len(closes)):
        if closes[i-1] > 0:
            rets.append((closes[i]/closes[i-1]) - 1.0)
    return rets


def _rolling_var_95(rets: List[float], window: int = 60) -> List[float]:
    out: List[float] = []
    if len(rets) < window:
        return out
    import numpy as np
    for i in range(window, len(rets)+1):
        w = rets[i-window:i]
        q = float(np.percentile(w, 5))
        out.append(abs(min(0.0, q)))
    return out


def _extract_financial_series(income_fin: dict):
    annual = []
    quarterly = []
    for blk in income_fin.get("raw") or []:
        data = blk.get("data") or []
        if isinstance(data, list):
            for row in data:
                r = row.get("revenue") or row.get("total_revenue")
                n = row.get("net_income") or row.get("netIncome")
                fy = row.get("fiscal_year") or row.get("fiscalYear") or row.get("year")
                fq = row.get("fiscal_quarter") or row.get("fiscalQuarter") or row.get("quarter")
                period = (row.get("period") or row.get("frequency") or "").lower()
                item = {"revenue": float(r) if isinstance(r, (int, float)) else None,
                        "net_income": float(n) if isinstance(n, (int, float)) else None}
                if fq or period == "quarterly":
                    if fy is not None and fq is not None:
                        try:
                            quarterly.append({"year": int(fy), "q": int(fq), **item})
                        except Exception:
                            pass
                else:
                    if fy is not None:
                        try:
                            annual.append({"year": int(fy), **item})
                        except Exception:
                            pass
    # sort
    annual.sort(key=lambda x: x.get("year", 0))
    quarterly.sort(key=lambda x: (x.get("year", 0), x.get("q", 0)))
    return annual, quarterly


def _growth_rates(series: List[float]) -> List[Optional[float]]:
    out: List[Optional[float]] = [None]
    for i in range(1, len(series)):
        prev = series[i-1]
        out.append(((series[i]/prev)-1.0) if prev else None)
    return out


def _yoy_quarterly(series: List[float]) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for i in range(len(series)):
        j = i-4
        if j >= 0 and series[j]:
            out.append((series[i]/series[j])-1.0)
        else:
            out.append(None)
    return out


def _render_charts(fig_dir: Path, instrument: str, closes: List[float], dates: List[str], income_fin: Optional[dict], target_weight: float, var_threshold: float = 0.035) -> List[str]:
    ensure_dir(fig_dir)
    paths: List[str] = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Price chart
        if closes:
            # Try to parse date strings to datetimes for x-axis
            from datetime import datetime
            import matplotlib.dates as mdates

            x_dates = []
            all_dt_ok = True
            for d in (dates or []):
                try:
                    x_dates.append(datetime.fromisoformat(str(d).replace("Z", "+00:00")))
                except Exception:
                    all_dt_ok = False
                    break
            if not all_dt_ok or (x_dates and len(x_dates) != len(closes)):
                x_dates = list(range(len(closes)))

            # Price
            plt.figure(figsize=(8, 3))
            plt.plot(x_dates, closes, label=f"{instrument} Close")
            plt.title(f"{instrument} Price")
            plt.xlabel("Date" if isinstance(x_dates[0], datetime) else "t")
            plt.ylabel("Price")
            if x_dates and isinstance(x_dates[0], datetime):
                ax = plt.gca()
                locator = mdates.AutoDateLocator()
                formatter = mdates.ConciseDateFormatter(locator)
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)
                plt.gcf().autofmt_xdate()
            plt.legend()
            p1 = fig_dir / "price.png"
            plt.tight_layout()
            plt.savefig(p1)
            plt.close()
            paths.append(str(p1))

            # Drawdown
            import numpy as np
            arr = np.array(closes)
            peak = np.maximum.accumulate(arr)
            dd = (arr - peak) / peak
            plt.figure(figsize=(8, 2.5))
            plt.plot(x_dates, dd, color="crimson")
            plt.title(f"{instrument} Drawdown")
            plt.xlabel("Date" if (x_dates and isinstance(x_dates[0], datetime)) else "t")
            plt.ylabel("Drawdown")
            if x_dates and isinstance(x_dates[0], datetime):
                ax = plt.gca()
                locator = mdates.AutoDateLocator()
                formatter = mdates.ConciseDateFormatter(locator)
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)
                plt.gcf().autofmt_xdate()
            p2 = fig_dir / "drawdown.png"
            plt.tight_layout()
            plt.savefig(p2)
            plt.close()
            paths.append(str(p2))

            # Rolling VaR curve (empirical 95% over 60d, scaled by target weight)
            rets = _compute_returns(closes)
            rv = _rolling_var_95(rets, window=60)
            if rv:
                scaled = [v * max(0.0, target_weight) for v in rv]
                # Align VaR points to dates from index 60 onward (len(closes) - len(rv)) should be 60
                if x_dates and not isinstance(x_dates[0], int):
                    try:
                        x_var = x_dates[60:60 + len(scaled)]
                        if len(x_var) != len(scaled):
                            x_var = list(range(len(scaled)))
                    except Exception:
                        x_var = list(range(len(scaled)))
                else:
                    x_var = list(range(len(scaled)))

                plt.figure(figsize=(8, 2.5))
                plt.plot(x_var, scaled, label="Rolling VaR@95 (60d)")
                plt.axhline(y=var_threshold, color="orange", linestyle="--", label=f"Threshold {var_threshold:.3%}")
                plt.title(f"{instrument} VaR Curve (scaled by weight)")
                plt.xlabel("Date" if (x_dates and not isinstance(x_dates[0], int) and isinstance(x_var[0], datetime)) else "t")
                plt.ylabel("VaR")
                if x_dates and not isinstance(x_dates[0], int) and isinstance(x_var[0], datetime):
                    ax = plt.gca()
                    locator = mdates.AutoDateLocator()
                    formatter = mdates.ConciseDateFormatter(locator)
                    ax.xaxis.set_major_locator(locator)
                    ax.xaxis.set_major_formatter(formatter)
                    plt.gcf().autofmt_xdate()
                plt.legend()
                pvar = fig_dir / "var_curve.png"
                plt.tight_layout()
                plt.savefig(pvar)
                plt.close()
                paths.append(str(pvar))

    # Income trend (very best-effort: look for revenue/net_income arrays)
        if income_fin:
            annual, quarterly = _extract_financial_series(income_fin)
            # Annual trend
            if annual:
                yrs = [a["year"] for a in annual]
                revs = [a.get("revenue") for a in annual]
                nis = [a.get("net_income") for a in annual]
                plt.figure(figsize=(8, 3))
                if any(v is not None for v in revs):
                    plt.plot(yrs, [v if v is not None else float('nan') for v in revs], marker='o', label="Revenue")
                if any(v is not None for v in nis):
                    plt.plot(yrs, [v if v is not None else float('nan') for v in nis], marker='o', label="Net Income")
                plt.title(f"{instrument} Annual Income Trend")
                plt.xlabel("Year")
                plt.ylabel("$")
                plt.xticks(yrs, rotation=0)
                plt.legend()
                p_ann = fig_dir / "income_trend_annual.png"
                plt.tight_layout()
                plt.savefig(p_ann)
                plt.close()
                paths.append(str(p_ann))

                # Annual YoY growth
                if len(yrs) >= 2:
                    import numpy as np
                    plt.figure(figsize=(8, 2.5))
                    if any(v is not None for v in revs):
                        ry = _growth_rates([v if v is not None else np.nan for v in revs])
                        plt.plot(yrs, [v if v is not None else float('nan') for v in ry], marker='o', label="Rev YoY")
                    if any(v is not None for v in nis):
                        ny = _growth_rates([v if v is not None else np.nan for v in nis])
                        plt.plot(yrs, [v if v is not None else float('nan') for v in ny], marker='o', label="NI YoY")
                    plt.axhline(0, color='gray', linewidth=0.8)
                    plt.title(f"{instrument} Annual YoY Growth")
                    plt.xlabel("Year")
                    plt.ylabel("Growth")
                    plt.legend()
                    p_ann_g = fig_dir / "income_growth_annual.png"
                    plt.tight_layout()
                    plt.savefig(p_ann_g)
                    plt.close()
                    paths.append(str(p_ann_g))

            # Quarterly trend
            if quarterly:
                labels = [f"{q['year']}Q{q['q']}" for q in quarterly]
                revs_q = [q.get("revenue") for q in quarterly]
                nis_q = [q.get("net_income") for q in quarterly]
                x = list(range(len(labels)))
                plt.figure(figsize=(10, 3))
                if any(v is not None for v in revs_q):
                    plt.plot(x, [v if v is not None else float('nan') for v in revs_q], marker='.', label="Revenue")
                if any(v is not None for v in nis_q):
                    plt.plot(x, [v if v is not None else float('nan') for v in nis_q], marker='.', label="Net Income")
                plt.title(f"{instrument} Quarterly Income Trend")
                plt.xlabel("Quarter")
                plt.ylabel("$")
                # show sparse ticks to avoid clutter
                step = max(1, len(labels)//8)
                plt.xticks(x[::step], [labels[i] for i in range(0, len(labels), step)], rotation=45)
                plt.legend()
                p_q = fig_dir / "income_trend_quarterly.png"
                plt.tight_layout()
                plt.savefig(p_q)
                plt.close()
                paths.append(str(p_q))

                # Quarterly QoQ and YoY growth for revenue
                import numpy as np
                if any(v is not None for v in revs_q):
                    rq = [v if v is not None else np.nan for v in revs_q]
                    qoq = _growth_rates(rq)
                    yoy = _yoy_quarterly(rq)
                    plt.figure(figsize=(10, 2.5))
                    plt.plot(x, [v if v is not None else float('nan') for v in qoq], label="QoQ")
                    plt.plot(x, [v if v is not None else float('nan') for v in yoy], label="YoY")
                    plt.axhline(0, color='gray', linewidth=0.8)
                    plt.title(f"{instrument} Revenue Growth (QoQ / YoY)")
                    plt.xlabel("Quarter")
                    # show sparse ticks
                    step = max(1, len(labels)//8)
                    plt.xticks(x[::step], [labels[i] for i in range(0, len(labels), step)], rotation=45)
                    plt.ylabel("Growth")
                    plt.legend()
                    p_qg = fig_dir / "income_growth_quarterly.png"
                    plt.tight_layout()
                    plt.savefig(p_qg)
                    plt.close()
                    paths.append(str(p_qg))
    except Exception:
        pass
    return paths


# Removed code-side MCP price/volume fetching; the agent is responsible for retrieving and reporting these.

def generate_report(case: InvestmentCase, advice: Optional[InvestmentAdvice], out_path: str, language: str = "en", use_mcp: bool = False) -> str:
    # Load MCP artifacts for the case (best-effort)
    root = Path("knowledge")
    market = _load_market_artifacts(root, case.instrumentId, case.asOf)

    # Derive 6M performance and summaries
    perf6m = None
    quote = ""
    news_top5: List[str] = []
    filings_top5: List[str] = []
    closes: List[float] = []
    dates: List[str] = []
    latest_px: Optional[float] = None
    latest_dt: Optional[str] = None
    day_chg_pct: Optional[float] = None
    if market.get("prices"):
        closes, dates = _extract_series_from_prices(market["prices"])
        perf6m = _six_month_performance(closes)
        if closes:
            latest_px = closes[-1]
            day_chg_pct = _compute_day_change_from_series(closes)
        if dates:
            latest_dt = _last_nonempty(dates)
    else:
        perf6m = _six_month_performance(closes)
    if market.get("news"):
        news_top5 = _summarize_news(market["news"], topn=5)
    if market.get("filings"):
        filings_top5 = _summarize_filings(market["filings"], topn=5)
    # Prefer the more recent of prices vs quote for latest snapshot
    qobj = market.get("quote")
    if qobj:
        try:
            from datetime import datetime
            q_time = qobj.get("time")
            q_price = qobj.get("price")
            q_pct = qobj.get("day_change_percent")
            dt_prices = None
            if latest_dt:
                try:
                    dt_prices = datetime.fromisoformat(latest_dt.replace("Z", "+00:00"))
                except Exception:
                    dt_prices = None
            dt_quote = None
            if isinstance(q_time, str):
                try:
                    dt_quote = datetime.fromisoformat(q_time.replace("Z", "+00:00"))
                except Exception:
                    dt_quote = None
            # If quote is newer than prices, or prices missing, prefer quote
            if (dt_quote and (dt_prices is None or dt_quote > dt_prices)) or (latest_px is None and q_price is not None):
                latest_px = q_price if isinstance(q_price, (int, float)) else latest_px
                latest_dt = q_time or latest_dt
                day_chg_pct = q_pct if isinstance(q_pct, (int, float)) else day_chg_pct
        except Exception:
            pass
        quote = _summarize_quote(qobj)

    # Render charts
    fig_dir = Path(out_path).parent / "figs"
    var_threshold = 0.035
    figs = _render_charts(fig_dir, case.instrumentId, closes, dates, market.get("financials_income"), case.proposal.target_weight, var_threshold=var_threshold)
    # Volume charts are not generated by code; the agent should reference ADV/volume context in text.

    # Identify VaR figure for Risk section
    risk_var_rel: Optional[str] = None
    fig_items: List[dict] = []
    for p in figs:
        name = Path(p).name
        rel = Path(p).relative_to(Path(out_path).parent).as_posix()
        fig_items.append({"name": name, "path": rel, "is_var": name == "var_curve.png"})
        if name == "var_curve.png":
            risk_var_rel = rel

    # Build references
    references = _extract_references(market, case.researchMemo.sources)

    # Instrument alias hints to help tool search resolve canonical symbol
    aliases: List[str] = [case.instrumentId]
    try:
        if advice and advice.instrumentId and advice.instrumentId not in aliases:
            aliases.append(advice.instrumentId)
    except Exception:
        pass
    # Simple heuristic hints for BYD HK

    # Placeholder for composed content (no tools here)
    llm_content: Optional[str] = None

    # Try LLM-composed report (no tools)
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.2)
    sys_msg = (
        "You are an experienced buy-side analyst. Compose a professional, concise Markdown investment report. "
        "Include these sections in order: Summary; Research Thesis; Portfolio Proposal; Risk & Compliance (embed VaR chart if provided, do not duplicate it in Charts); "
        "Trading Plan; Advice Details (only if advice exists); Market Context (MCP); Charts (embed provided images except the VaR image); "
        "Methodology / Process; References (bullet list of clickable links). "
        "Write in the target language. Do not fabricate data or links. Do not use placeholders such as '??'. Return ONLY Markdown (no code fences)."
    )
    # Add explicit clarity guidance for the Portfolio Proposal wording
    sys_msg += (
        " In the Portfolio Proposal section, use clear, non-jargon wording: convert decimals to percentages; explain the build schedule as phased entries by date; "
        "define liquidity terms (adv_multiple = daily participation vs. ADV; days_to_build = days to finish building); and state constraints as percentages with a one-line reason."
    )
    # Price & Forecast section should reflect data present in case/advice; do not instruct tool usage here.
    sys_msg += (
        " Include a 'Price & Forecast' section that summarizes current price context, 30/90-day ADV and forecasts if present in the inputs; if unavailable, state 'N/A'. Use the provided 'market_summaries' fields for latest price/time and daily change; do not invent numbers."
    )
    sys_msg += (
        " In Risk & Compliance, explicitly state the VaR horizon and basis as 'VaR(95%, 1-day, portfolio-level, scaled by target weight)'."
    )
    sys_msg += (
        " In Advice Details, clarify that stop_loss/take_profit are multipliers relative to entry price (e.g., 0.9 = -10%, 1.2 = +20%)."
        " If position_size is provided, define active weight as the deviation from target (active_weight = position_size - target_weight), expressed in percentage points."
        " Do NOT multiply position_size by target_weight."
    )
    # Clarify horizon vs build schedule and adv_multiple wording to avoid mislabeling
    sys_msg += (
        " Do NOT describe advice.horizon_days as a build period; it is an evaluation/holding horizon."
        " Keep 'days_to_build' strictly for the execution window to reach target weight."
        " Explain adv_multiple as a cumulative participation guideline over the build window (not a single-day share >100%)."
        " If a current position exists (position_size) and differs from target_weight, explicitly state whether the schedule is to increase or decrease exposure accordingly."
    )
    human_msg = (
        "Target language: {language}\n\n"
        "Context JSON (case/advice/market summaries):\n{ctx}\n\n"
        "Charts to embed (Markdown image links):\n{charts}\n\n"
        "VaR threshold: {var_threshold} (use in text if relevant)."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_msg),
        ("human", human_msg),
    ])
    ctx = {
        "case": case.model_dump(),
        "advice": (advice.model_dump() if advice else None),
        "market_summaries": {
            "latest_price_text": quote,
            "latest_price": latest_px,
            "latest_time": latest_dt,
            "day_change_percent": day_chg_pct,
            "adv_30": (market.get("index") or {}).get("adv_30") if isinstance(market.get("index"), dict) else None,
            "adv_90": (market.get("index") or {}).get("adv_90") if isinstance(market.get("index"), dict) else None,
            "news": news_top5,
            "filings": filings_top5,
            "perf6m": perf6m,
            "perf6m_asof": (latest_dt or case.asOf),
        },
        "active_weight": ((advice.position_size - case.proposal.target_weight) if (advice and advice.position_size is not None) else None),
        "existing_position_size": (advice.position_size if advice and advice.position_size is not None else None),
        "position_delta": ((case.proposal.target_weight - advice.position_size) if (advice and advice.position_size is not None) else None),
        "references": references,
        "risk_var_image": risk_var_rel,
    }
    charts_md = []
    for f in fig_items:
        if not f.get("is_var"):
            charts_md.append(f"- {f['name']}: ![{f['name']}]({f['path']})")
    chain = prompt | llm
    llm_raw = chain.invoke({
        "language": language,
        "ctx": json.dumps(ctx, ensure_ascii=False),
        "charts": "\n".join(charts_md),
        "var_threshold": f"{var_threshold:.1%}",
    })
    llm_content = getattr(llm_raw, "content", llm_raw)
    if not isinstance(llm_content, str):
        llm_content = None

    # Ensure References section exists; append if missing
    content = llm_content
    # Do not inject Price & Forecast via code; the agent must produce it via tools per the prompt.
    if references and ("\n## References" not in content and "\n# References" not in content):
        content = content.rstrip() + "\n\n## References\n" + "\n".join(references) + "\n"
    # Optionally ensure Charts are present
    if fig_items and ("\n## Charts" not in content and "![](" not in content and "![" not in content):
        charts_block = ["## Charts"] + [f"![{f['name']}]({f['path']})" for f in fig_items if not f.get("is_var")]
        content = content.rstrip() + "\n\n" + "\n".join(charts_block) + "\n"

    write_text(out_path, content)
    return out_path
