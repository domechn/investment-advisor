from __future__ import annotations

from datetime import date
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Scenario(BaseModel):
    fv: float = Field(..., description="Fair value estimate for the scenario")
    triggers: List[str] = Field(default_factory=list)


class Valuation(BaseModel):
    method: str
    wacc: Optional[float] = None
    scenarios: Dict[str, Scenario]


class ResearchMemo(BaseModel):
    ticker_symbol: str
    thesis: str
    antithesis: str
    moat_assessment: Dict[str, object] = Field(default_factory=dict)
    unit_economics: Dict[str, float] = Field(default_factory=dict)
    valuation: Valuation
    risks: List[str] = Field(default_factory=list)
    exit_conditions: List[str] = Field(default_factory=list)
    sources: List[Dict[str, str]] = Field(default_factory=list)


class BuildScheduleItem(BaseModel):
    date: str
    pct: float


class Proposal(BaseModel):
    target_weight: float
    build_schedule: List[BuildScheduleItem] = Field(default_factory=list)
    liquidity: Dict[str, float] = Field(default_factory=dict)
    constraints: Dict[str, float] = Field(default_factory=dict)


class RiskCheck(BaseModel):
    var_95: float
    max_drawdown: Optional[float] = None
    breach: bool = False
    notes: Optional[str] = None


class ComplianceCheck(BaseModel):
    restricted: bool
    notes: Optional[str] = None


class TradingPlan(BaseModel):
    algo: str = "TWAP"
    tca_benchmark: str = "VWAP"
    expected_cost_bps: float = 12.0


class Decision(BaseModel):
    status: Literal["IC_APPROVED", "IC_REJECTED", "NEEDS_REVISION"]
    approvers: List[str] = Field(default_factory=list)
    timestamp: Optional[str] = None


class InvestmentCase(BaseModel):
    instrumentId: str
    asOf: str
    researchMemo: ResearchMemo
    proposal: Proposal
    riskCheck: RiskCheck
    complianceCheck: ComplianceCheck
    tradingPlan: TradingPlan
    decision: Decision


class InvestmentAdvice(BaseModel):
    instrumentId: str
    asOf: str
    recommendation: Literal["BUY", "HOLD", "SELL"]
    target_price: Optional[float] = None
    position_size: Optional[float] = Field(None, description="Target portfolio weight, 0-1")
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    horizon_days: Optional[int] = None
    rationale: str
    key_risks: List[str] = Field(default_factory=list)
    catalysts: List[str] = Field(default_factory=list)
    market_context: Dict[str, object] = Field(default_factory=dict)
