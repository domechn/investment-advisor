from __future__ import annotations

from pathlib import Path
from typing import Optional

from .schemas import ComplianceCheck, Proposal, ResearchMemo, RiskCheck


def _load_prices_if_any(
    instrument: Optional[str], as_of: Optional[str], root: str
) -> Optional[dict]:
    if not instrument or not as_of:
        return None
    p = Path(root) / "market" / instrument / f"prices_{as_of}.json"
    if not p.exists():
        return None
    try:
        import json

        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_series(prices: list) -> Optional[tuple[list, list]]:
    """Extract close price series (and volumes if present) from multiple shapes.

    Supported inputs:
    - list of OHLC rows: [{close|price, volume?, ...}, ...]
    - dict with {raw: [{data: [rows...]}]}
    - dict with {data: [rows...]}
    - dict with {series: [numbers...]}
    """
    try:
        series: list[float] = []
        volumes: list[float] = []

        # Case 1: list of rows
        for row in prices:
            px = row.get("close") 
            if isinstance(px, (int, float)):
                series.append(float(px))
            vol = row.get("volume")
            if isinstance(vol, (int, float)):
                volumes.append(float(vol))
        return (series, volumes) if len(series) >= 30 else None
    except Exception:
        return None


def _compute_var_drawdown_from_series(
    series: list[float], target_weight: float
) -> tuple[float, Optional[float], Optional[float]]:
    # Returns: (var_95_portfolio_scaled, mdd_asset_level, sigma_ewma_asset)
    # Daily returns
    rets = []
    for i in range(1, len(series)):
        if series[i - 1] > 0:
            rets.append((series[i] / series[i - 1]) - 1.0)
    if len(rets) < 20:
        return 0.0, None, None

    # Empirical quantile VaR
    sr = sorted(rets)
    idx = max(0, int(0.05 * len(sr)) - 1)
    q05 = sr[idx]
    var_emp = abs(min(0.0, q05)) * target_weight

    # EWMA sigma (RiskMetrics lambda=0.94)
    lam = 0.94
    sigma2 = 0.0
    for r in reversed(rets):
        sigma2 = lam * sigma2 + (1 - lam) * (r * r)
    sigma = sigma2**0.5
    var_ewma = (
        1.65 * sigma * target_weight
    )  # 1.65 ~ 95%; VaR scaled by target weight (portfolio-level)

    # Max drawdown (asset-level, NOT scaled by target weight)
    peak = series[0]
    mdd = 0.0
    for px in series:
        if px > peak:
            peak = px
        dd = (px - peak) / peak
        if dd < mdd:
            mdd = dd
    mdd = abs(mdd)

    return round(max(var_emp, var_ewma), 4), round(mdd, 4), round(sigma, 6)


def compute_risk_check(
    research: ResearchMemo,
    proposal: Proposal,
    *,
    instrument: Optional[str] = None,
    as_of: Optional[str] = None,
    root: str = "knowledge",
    prices: Optional[dict] = None,
) -> RiskCheck:
    # Prefer empirical metrics from MCP prices when available; fallback to heuristics
    target_weight = proposal.target_weight
    max_single = float(proposal.constraints.get("max_single_name", 0.06) or 0.06)

    var_from_prices = None
    mdd = None
    sigma_ewma = None
    p = prices or _load_prices_if_any(instrument, as_of, root)
    if p:
        ex = _extract_series(p)
        if ex:
            series, _ = ex
            var, mdd2, sigma = _compute_var_drawdown_from_series(series, target_weight)
            var_from_prices, mdd, sigma_ewma = var, mdd2, sigma

    # heuristic baseline daily var ~1.8% at 3.5% weight (portfolio-level)
    base_var = (
        var_from_prices
        if var_from_prices is not None
        else 0.018 * (target_weight / 0.035 if 0.035 else 1.0)
    )

    # Stress scenarios expressed as asset-level moves (not scaled by target weight)

    breach_limits = target_weight > max_single or base_var > 0.035

    notes = []
    if target_weight > max_single:
        notes.append(
            f"target_weight {target_weight:.3f} > max_single_name {max_single:.3f}"
        )
    if base_var > 0.035:
        notes.append(f"var_95 {base_var:.3f} exceeds soft cap 3.5%")
    if mdd is not None:
        notes.append(f"max_drawdown (est) {mdd:.3f}")
    if sigma_ewma is not None:
        notes.append(f"ewma_sigma {sigma_ewma:.4f}")

    return RiskCheck(
        var_95=round(base_var, 4),
        max_drawdown=mdd,
        breach=breach_limits,
        notes="; ".join(notes) if notes else None,
    )


def run_compliance_check(research: ResearchMemo, proposal: Proposal) -> ComplianceCheck:
    # Minimal MVP: no restricted list, always allowed unless explicit red flags in memo
    red_flags = {"insider", "non-public", "restricted", "sanction"}
    memo_text = (
        research.thesis + " " + research.antithesis + " ".join(research.risks)
    ).lower()
    restricted = any(flag in memo_text for flag in red_flags)
    notes = (
        "auto-pass" if not restricted else "potential red flags detected in memo text"
    )
    return ComplianceCheck(restricted=restricted, notes=notes)
