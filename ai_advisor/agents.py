from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Optional, List, Dict, Any

from .schemas import Proposal, ResearchMemo, TradingPlan
from .policy import load_policy_from_env
import os
from .mcp_client import get_mcp_client, wrap_mcp_tools_for_logging
from .logging_callbacks import ToolLogHandler


def _enforce_policy_limits(proposal: Proposal, context: Optional[Dict[str, Any]] = None) -> Proposal:
    """Clamp proposal to firm policy baselines and align schedule with existing position if provided.

    Policy baselines (can later be made configurable):
    - max_single_name <= 0.06
    - sector_cap <= 0.20
    - target_weight <= max_single_name
    - If existing_position_size is provided in context, ensure build_schedule sums to delta (target - existing)
      with matching sign. If current schedule mismatches or is empty, rebuild a 2-step schedule.
    """
    try:
        policy = load_policy_from_env()
        constraints: Dict[str, float] = dict(proposal.constraints or {})
        firm_max_single = float(getattr(policy, "max_single_name", 0.06))
        firm_sector_cap = float(getattr(policy, "sector_cap", 0.20))
        if not constraints:
            constraints = {}
        # Clamp constraints
        ms = float(constraints.get("max_single_name", firm_max_single) or firm_max_single)
        sc = float(constraints.get("sector_cap", firm_sector_cap) or firm_sector_cap)
        ms = min(ms, firm_max_single)
        sc = min(sc, firm_sector_cap)
        constraints["max_single_name"] = ms
        constraints["sector_cap"] = sc
        proposal.constraints = constraints

        # Clamp target to max_single_name
        if proposal.target_weight > ms:
            proposal.target_weight = ms

        # Align schedule to bridge existing -> target, if existing provided
        existing = None
        try:
            if isinstance(context, dict):
                existing = context.get("existing_position_size")
        except Exception:
            existing = None
        if isinstance(existing, (int, float)):
            need = float(proposal.target_weight) - float(existing)
            # Compute current sum
            sched = list(proposal.build_schedule or [])
            current_sum = 0.0
            for it in sched:
                try:
                    current_sum += float(getattr(it, "pct", 0.0))
                except Exception:
                    pass
            # If mismatch in sign or magnitude (>10% relative error), rebuild a simple 2-step schedule
            def _rebuild(delta: float):
                from .schemas import BuildScheduleItem
                if abs(delta) < 1e-6:
                    return []
                half = delta / 2.0
                # Two roughly equal steps one month apart from today context not known here; keep incoming dates if any, else use placeholders
                # Keep dates unspecified at this layer; callers typically provide dates via LLM. Here we fallback to generic monthly anchors.
                from datetime import date, timedelta
                d1 = (date.today() + timedelta(days=14)).isoformat()
                d2 = (date.today() + timedelta(days=45)).isoformat()
                return [BuildScheduleItem(date=d1, pct=round(half, 6)), BuildScheduleItem(date=d2, pct=round(delta - half, 6))]

            def _rescale(delta: float, items: List[Any]):
                total = sum(float(getattr(x, "pct", 0.0)) for x in items) or 0.0
                if total == 0.0:
                    return _rebuild(delta)
                scale = delta / total
                out = []
                for it in items:
                    try:
                        it.pct = round(float(it.pct) * scale, 6)
                        out.append(it)
                    except Exception:
                        pass
                return out

            if (need == 0.0 and current_sum != 0.0) or (need * current_sum < 0) or (abs(current_sum - need) > max(0.0005, 0.1 * abs(need))):
                # Try rescaling if same sign; otherwise rebuild
                if current_sum != 0.0 and (need * current_sum) > 0:
                    proposal.build_schedule = _rescale(need, sched)
                else:
                    proposal.build_schedule = _rebuild(need)
    except Exception:
        # Best-effort enforcement; do not crash pipeline
        return proposal
    return proposal

def _make_llm(model: Optional[str] = None):
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError("LangChain/OpenAI not installed. Install requirements.txt") from e
    model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model_name, temperature=0.2)


async def generate_research_memo(idea: str, model: Optional[str] = None, language: str = "en", use_mcp: bool = False) -> ResearchMemo:
    # If MCP tools are enabled, run two specialized agents (financial-only and web-search) and merge.
    if use_mcp:
        try:
            from langchain_openai import ChatOpenAI  # type: ignore
            from langgraph.prebuilt import create_react_agent  # type: ignore
        except Exception:
            use_mcp = False
        else:
            # Fetch all tools and split into two toolsets
            try:
                client = get_mcp_client()
                tools_all = await client.get_tools()
            except Exception:
                tools_all = []

            def _by_name(name: str) -> Any:
                for t in tools_all:
                    if getattr(t, "name", "") == name:
                        return t
                return None

            # Allow lists based on observed tool names
            financial_allow: List[str] = [
                "get_current_stock_price",
                "get_historical_stock_prices",
                "get_income_statements",
                "get_balance_sheets",
                "get_cash_flow_statements",
                "get_sec_filings",
                "get_company_news",
                # crypto not needed normally, but harmless
                "get_current_crypto_price",
                "get_crypto_prices",
                "get_historical_crypto_prices",
            ]
            search_allow: List[str] = [
                "google_search",
                "extract_webpage_content",
                "extract_multiple_webpages",
            ]

            tools_fin = [t for t in tools_all if getattr(t, "name", "") in financial_allow]
            tools_web = [t for t in tools_all if getattr(t, "name", "") in search_allow]

            # Enable logging and ensure tools are ready for async usage
            if os.getenv("MCP_LOG"):
                import logging
                log = logging.getLogger("mcp.tools")
                tools_fin = wrap_mcp_tools_for_logging(tools_fin, log)
                tools_web = wrap_mcp_tools_for_logging(tools_web, log)

            llm_tools = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.2)
            callbacks = None
            if os.getenv("MCP_LOG"):
                import logging
                callbacks = [ToolLogHandler(logging.getLogger("mcp.tools"))]

            # Build two agents
            fin_agent = create_react_agent(llm_tools, tools_fin)
            web_agent = create_react_agent(llm_tools, tools_web)

            response_json_fields = (
                "ticker_symbol, thesis, antithesis, moat_assessment {type,evidence[]}, "
                "unit_economics {lt_margins, roe}, valuation {method, wacc, scenarios {bear|base|bull {fv, triggers[]}}}, "
                "risks[], exit_conditions[]"
            )

            # Prompts
            fin_sys = (
                "You are a senior equity research analyst restricted to FINANCIAL DATA tools provided. "
                "Use ONLY the available financial tools to gather official filings (annual/quarterly), press releases, prices, "
                "and reputable news about the company inferred from the idea. Do NOT call web search or webpage extraction. "
                "Do NOT open or summarize PDFs. If a filing is PDF-only, ignore contents. "
                "BEFORE completing, you MUST fetch the latest quarterly or annual earnings, or an official press release within the last 30 days if available, and reflect revenue, gross margin, operating margin, net income, and EPS with YoY/QoQ if possible. "
                "Also derive a brief 7-day sentiment label and score from recent headlines (no sources field required here). "
                f"Return ONLY JSON with fields: {response_json_fields}. DO NOT include a 'sources' field."
            )
            fin_user = (
                "Language: {language}\n"
                "Idea: {idea}\n"
                "Task: Use financial tools only and output the JSON (no sources)."
            )

            web_sys = (
                "You are a research assistant restricted to GOOGLE SEARCH tools. "
                "Use google_search to discover official links and credible coverage. "
                "You MAY extract a page only if it's HTML; NEVER fetch or summarize PDFs. "
                "Find and include links for the latest earnings/press releases and key data used, but DO NOT compute new numbers yourself. "
                f"Return ONLY JSON with fields: {response_json_fields}, and include sources[] as URLs you found. "
                "Do NOT invent links."
            )
            web_user = (
                "Language: {language}\n"
                "Idea: {idea}\n"
                "Task: Use google-search (and HTML extraction if needed) to assemble a ResearchMemo JSON, including sources."
            )

            async def _invoke_json(agent, sys_prompt: str, user_prompt: str) -> Dict[str, Any]:
                cfg = {"callbacks": callbacks} if callbacks else None
                # Use async invocation to support tools that are async-only (StructuredTool)
                res = await agent.ainvoke({
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                }, config=cfg)
                msgs = res.get("messages") if isinstance(res, dict) else None
                content = None
                if isinstance(msgs, list) and msgs:
                    last = msgs[-1]
                    content = getattr(last, "content", None) or (last.get("content") if isinstance(last, dict) else None)
                if not content and isinstance(res, dict):
                    content = res.get("final")
                if isinstance(content, list):
                    content = "".join(x if isinstance(x, str) else str(x) for x in content)
                if not isinstance(content, str):
                    raise ValueError("No text content from agent")
                m = re.search(r"\{[\s\S]*\}", content)
                if not m:
                    raise ValueError("No JSON found in agent output")
                return json.loads(m.group(0))

            def _coalesce(a, b):
                return a if (a is not None and a != "" and a != []) else b

            def _merge(fin: Dict[str, Any], web: Dict[str, Any]) -> Dict[str, Any]:
                out: Dict[str, Any] = {}
                # simple scalars/objects prefer financial
                out["thesis"] = _coalesce(fin.get("thesis"), web.get("thesis"))
                out["antithesis"] = _coalesce(fin.get("antithesis"), web.get("antithesis"))
                # moat_assessment merge evidence
                moat_fin = fin.get("moat_assessment") or {}
                moat_web = web.get("moat_assessment") or {}
                out["moat_assessment"] = {
                    "type": _coalesce(moat_fin.get("type"), moat_web.get("type")),
                    "evidence": list({*(moat_fin.get("evidence") or []), *(moat_web.get("evidence") or [])})
                }
                # unit economics
                out["unit_economics"] = _coalesce(fin.get("unit_economics"), web.get("unit_economics"))
                # valuation prefer financial wholesale
                val = fin.get("valuation") or web.get("valuation") or {}
                out["valuation"] = val
                # risks and exit_conditions: union
                def _uniqlist(lst: Any) -> List[Any]:
                    arr = lst if isinstance(lst, list) else []
                    seen = set()
                    uniq = []
                    for x in arr:
                        if x not in seen:
                            seen.add(x)
                            uniq.append(x)
                    return uniq
                out["risks"] = _uniqlist((fin.get("risks") or []) + (web.get("risks") or []))
                out["exit_conditions"] = _uniqlist((fin.get("exit_conditions") or []) + (web.get("exit_conditions") or []))
                # sources only from web
                srcs = web.get("sources") or []
                # dedupe by url
                seen = set()
                src_out = []
                for s in srcs:
                    try:
                        u = s.get("url")
                        if u and u not in seen:
                            seen.add(u)
                            src_out.append({
                                "type": s.get("type") or "web",
                                "url": u,
                                **({"title": s.get("title")} if s.get("title") else {}),
                            })
                    except Exception:
                        continue
                out["sources"] = src_out
                return out

            try:
                fin_json = await _invoke_json(fin_agent, fin_sys, fin_user.format(language=language, idea=idea))
            except Exception:
                fin_json = {}
            try:
                web_json = await _invoke_json(web_agent, web_sys, web_user.format(language=language, idea=idea))
            except Exception:
                web_json = {}
            try:
                merged = _merge(fin_json, web_json)
                return ResearchMemo(**merged)
            except Exception:
                # If merging/validation fails, fall through to non-tool path
                use_mcp = False
    # Lazy import prompt template
    try:
        from langchain_core.prompts import ChatPromptTemplate  # type: ignore
    except Exception as e:
        raise RuntimeError("LangChain not installed. Install requirements.txt") from e
    llm = _make_llm(model)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a senior equity research analyst with a value-investing mindset. Output must strictly match the target Pydantic schema. Use the target language: {language}"),
        ("human", "Idea: {idea}\nGenerate a concise yet complete research memo including thesis, antithesis, moat assessment, unit economics, valuation with three scenarios (bear/base/bull), key risks, exit conditions, and cite sources. Keep numbers reasonable and consistent.")
    ])
    structured_llm = llm.with_structured_output(ResearchMemo, method="function_calling")
    chain = prompt | structured_llm
    try:
        return chain.invoke({"idea": idea, "language": language})
    except Exception:
        # Fallback to strict-JSON prompt and manual validation (no mock)
        from langchain_core.prompts import ChatPromptTemplate  # type: ignore
        llm_fallback = _make_llm(model)
        json_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You must output ONLY a JSON object for the ResearchMemo schema. Required fields: "
                "ticker_symbol (string), thesis (string), antithesis (string), moat_assessment (object with keys: type (string), evidence (array of strings)), "
                "unit_economics (object with keys: lt_margins (number), roe (number)), "
                "valuation (object with keys: method (string), wacc (number or null), scenarios (object containing keys 'bear', 'base', 'bull'; each is an object with keys: fv (number), triggers (array of strings))), "
                "risks (array of strings), exit_conditions (array of strings), sources (array of objects with keys: type (string), url (string), title (optional string)). "
                "No prose, no bullets, no markdown. Return valid JSON only."
            ),
            ("human", "Language: {language}\nIdea: {idea}\nReturn valid JSON only.")
        ])
        resp = (json_prompt | llm_fallback).invoke({"idea": idea, "language": language})
        text = getattr(resp, "content", resp)
        if not isinstance(text, str):
            raise RuntimeError("Empty response for ResearchMemo JSON fallback")
        m = re.search(r"\{[\s\S]*\}", text)
        data = json.loads(m.group(0) if m else text)
    return ResearchMemo(**data)


async def propose_portfolio(research_memo: ResearchMemo, model: Optional[str] = None, context: Optional[dict] = None, language: str = "en", use_mcp: bool = False) -> Proposal:
    # If MCP tools are enabled, use a REACT agent with tools to let it fetch prices/financials as needed
    if use_mcp:
        try:
            from langchain_openai import ChatOpenAI  # type: ignore
            from langgraph.prebuilt import create_react_agent  # type: ignore
        except Exception:
            use_mcp = False
        else:
            try:
                client = get_mcp_client()
                tools = await client.get_tools()
            except Exception:
                tools = []
            llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.2)
            callbacks = None
            if os.getenv("MCP_LOG"):
                import logging
                callbacks = [ToolLogHandler(logging.getLogger("mcp.tools"))]
            # Log tool calls if enabled and keep tools async-capable
            if os.getenv("MCP_LOG"):
                import logging
                tools = wrap_mcp_tools_for_logging(tools, logging.getLogger("mcp.tools"))
            agent = create_react_agent(llm, tools)
            user_msg = (
                "You are a portfolio manager. Use tools if helpful (prices for ADV/volatility, financials for scale). "
                "Return ONLY a JSON for Proposal with keys: target_weight (<=0.06), build_schedule (array of {date,pct}), "
                "liquidity (adv_multiple, days_to_build), constraints (max_single_name, sector_cap). "
                "If Context.existing_position_size is provided, ensure build_schedule direction reflects moving from existing position to target_weight: "
                "if existing > target, produce a DECREASE schedule (negative pct or descriptive reductions); if existing < target, produce an INCREASE schedule. "
                "The cumulative schedule should bridge the gap between existing and target within constraints; avoid overshooting max_single_name. "
                f"Language: {language}. Memo: {research_memo.model_dump_json()} Context: {(context or {})}"
            )
            try:
                invoke_cfg = {"callbacks": callbacks} if callbacks else None
                # Use synchronous invoke to prevent event loop churn
                # Async path to avoid sync-only invocation of async tools
                result = await agent.ainvoke({"messages": user_msg}, config=invoke_cfg)
                messages = result.get("messages") if isinstance(result, dict) else None
                content = None
                if isinstance(messages, list) and messages:
                    last = messages[-1]
                    content = getattr(last, "content", None) or (last.get("content") if isinstance(last, dict) else None)
                if not content and isinstance(result, dict):
                    content = result.get("final", None)
                if isinstance(content, list):
                    content = "".join(x if isinstance(x, str) else str(x) for x in content)
                if not isinstance(content, str):
                    raise ValueError("No text content from agent")
                m = re.search(r"\{[\s\S]*\}", content)
                if not m:
                    raise ValueError("No JSON found in agent output")
                data = json.loads(m.group(0))
                proposal = Proposal(**data)
                # Normalize days_to_build vs build_schedule when using tools
                try:
                    sched = proposal.build_schedule or []
                    liq = proposal.liquidity or {}
                    if sched and isinstance(liq, dict):
                        from datetime import datetime
                        fmt_candidates = ["%Y-%m-%d", "%Y/%m/%d"]
                        def _parse(d: str):
                            for f in fmt_candidates:
                                try:
                                    return datetime.strptime(d, f)
                                except Exception:
                                    continue
                            return None
                        ds = [_parse(it.date) for it in sched if getattr(it, "date", None)]
                        ds = [d for d in ds if d is not None]
                        if ds:
                            days = (max(ds) - min(ds)).days or 1
                            db = liq.get("days_to_build")
                            if not isinstance(db, (int, float)) or abs(float(db) - float(days)) > 10:
                                liq["days_to_build"] = float(days)
                                proposal.liquidity = liq
                except Exception:
                    pass
                # Enforce policy limits and align schedule to existing position if provided in context
                try:
                    proposal = _enforce_policy_limits(proposal, context)
                except Exception:
                    pass
                return proposal
            except Exception:
                # fall through to non-tool path
                use_mcp = False
    # Non-tool path: plain structured LLM
    try:
        from langchain_core.prompts import ChatPromptTemplate  # type: ignore
    except Exception as e:
        raise RuntimeError("LangChain not installed. Install requirements.txt") from e
    llm = _make_llm(model)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a portfolio manager focused on prudent concentration and liquidity-aware execution. Output must match the Pydantic schema. Use the target language: {language}"),
        ("human", "Given the following research memo, propose target weight (<=6%), a 2-3 step build schedule, liquidity assumptions (adv_multiple, days_to_build), and constraints (max_single_name, sector_cap). If market/financial data context is provided, use it prudently. If Context.existing_position_size is provided, set the schedule direction to move from existing to target (decrease if existing>target; increase if existing<target), and ensure cumulative steps bridge the gap within constraints.\nResearch memo: {memo}\nContext: {context}")
    ])
    structured_llm = llm.with_structured_output(Proposal, method="function_calling")
    chain = prompt | structured_llm
    proposal: Proposal = chain.invoke({"memo": research_memo.model_dump_json(), "context": (context or {}), "language": language})
    # Normalize days_to_build vs build_schedule if possible
    try:
        sched = proposal.build_schedule or []
        liq = proposal.liquidity or {}
        if sched and isinstance(liq, dict):
            from datetime import datetime
            fmt_candidates = ["%Y-%m-%d", "%Y/%m/%d"]
            def _parse(d: str) -> Optional[datetime]:
                for f in fmt_candidates:
                    try:
                        return datetime.strptime(d, f)
                    except Exception:
                        continue
                return None
            ds = [_parse(it.date) for it in sched if getattr(it, "date", None)]
            ds = [d for d in ds if d is not None]
            if ds:
                days = (max(ds) - min(ds)).days or 1
                db = liq.get("days_to_build")
                # If divergence > 10 days, align liquidity to schedule span
                if not isinstance(db, (int, float)) or abs(float(db) - float(days)) > 10:
                    liq["days_to_build"] = float(days)
                    proposal.liquidity = liq
    except Exception:
        pass
    # Policy enforcement
    try:
        proposal = _enforce_policy_limits(proposal, context)
    except Exception:
        pass
    return proposal


async def plan_trading(research_memo: ResearchMemo, proposal: Proposal, model: Optional[str] = None, context: Optional[dict] = None, language: str = "en", use_mcp: bool = False) -> TradingPlan:
    if use_mcp:
        try:
            from langchain_openai import ChatOpenAI  # type: ignore
            from langgraph.prebuilt import create_react_agent  # type: ignore
        except Exception:
            use_mcp = False
        else:
            try:
                client = get_mcp_client()
                tools = await client.get_tools()
            except Exception:
                tools = []
            llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.2)
            callbacks = None
            if os.getenv("MCP_LOG"):
                import logging
                callbacks = [ToolLogHandler(logging.getLogger("mcp.tools"))]
            # Log tool calls if enabled and keep tools async-capable
            if os.getenv("MCP_LOG"):
                import logging
                tools = wrap_mcp_tools_for_logging(tools, logging.getLogger("mcp.tools"))
            agent = create_react_agent(llm, tools)
            user_msg = (
                "You are a trading strategist. Use tools if needed (prices for ADV/volatility). "
                "Return ONLY a JSON for TradingPlan with keys: algo (TWAP|VWAP|POV), tca_benchmark, expected_cost_bps (10-30). "
                f"Language: {language}. Memo: {research_memo.model_dump_json()} Proposal: {proposal.model_dump_json()} Context: {(context or {})}"
            )
            try:
                invoke_cfg = {"callbacks": callbacks} if callbacks else None
                # Use synchronous invoke to prevent event loop churn
                # Async path to avoid sync-only invocation of async tools
                result = await agent.ainvoke({"messages": user_msg}, config=invoke_cfg)
                messages = result.get("messages") if isinstance(result, dict) else None
                content = None
                if isinstance(messages, list) and messages:
                    last = messages[-1]
                    content = getattr(last, "content", None) or (last.get("content") if isinstance(last, dict) else None)
                if not content and isinstance(result, dict):
                    content = result.get("final", None)
                if isinstance(content, list):
                    content = "".join(x if isinstance(x, str) else str(x) for x in content)
                if not isinstance(content, str):
                    raise ValueError("No text content from agent")
                m = re.search(r"\{[\s\S]*\}", content)
                if not m:
                    raise ValueError("No JSON found in agent output")
                data = json.loads(m.group(0))
                return TradingPlan(**data)
            except Exception:
                use_mcp = False
    try:
        from langchain_core.prompts import ChatPromptTemplate  # type: ignore
    except Exception as e:
        raise RuntimeError("LangChain not installed. Install requirements.txt") from e
    llm = _make_llm(model)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a buy-side trading strategist optimizing for low market impact and good fill quality. Use the target language: {language}"),
        ("human", "Given memo and proposal, select a simple algo (TWAP/VWAP/POV), a TCA benchmark, and estimate expected cost in bps (10-30). If market context (ADV/volatility) is available, reflect it conservatively.\nMemo: {memo}\nProposal: {proposal}\nContext: {context}")
    ])
    structured_llm = llm.with_structured_output(TradingPlan, method="function_calling")
    chain = prompt | structured_llm
    return chain.invoke({"memo": research_memo.model_dump_json(), "proposal": proposal.model_dump_json(), "context": (context or {}), "language": language})
