from __future__ import annotations

import asyncio
import json
import re
from datetime import date
from typing import Optional

import os
from .mcp_client import MCPNotAvailable, get_mcp_client
from .logging_callbacks import ToolLogHandler
from .schemas import InvestmentAdvice


def _build_market_context(ticker: str, use_mcp: bool) -> dict:
    # With REACT agent usage, we don't pre-fetch context here anymore.
    return {}


async def generate_investment_advice(ticker: str, *, use_mcp: bool = True, language: str = "en") -> InvestmentAdvice:
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
        from langgraph.prebuilt import create_react_agent  # type: ignore
    except Exception:
        raise RuntimeError("LangChain and LangGraph are required")

    as_of = date.today().isoformat()
    if use_mcp:
        # Build REACT agent with MCP tools. Let the agent decide which tools to call.
        try:
            client = get_mcp_client()
            tools = await client.get_tools()
        except Exception:
            tools = []
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        callbacks = None
        if os.getenv("MCP_LOG"):
            import logging
            callbacks = [ToolLogHandler(logging.getLogger("mcp.tools"))]
        agent = create_react_agent(llm, tools)
        # Instruct to use tools as needed and output a strict JSON per schema
        user_msg = (
            f"You are a disciplined buy-side PM. Analyze {ticker} using tools (news, prices, financials, filings, insider, ownership, crypto if relevant). "
            f"Prefer the financial data MCP. You MAY use google-search ONLY to find official links or credible coverage when the financial MCP lacks the data. Do not fetch or summarize large PDFs; just cite the link. "
            f"Compute MARKET SENTIMENT from news headlines in the last 7 days: classify as Positive/Neutral/Negative and estimate a score [-1,1]. Include 3-5 representative headlines used. "
            f"Be conservative on sizing vs. volatility/liquidity. Return ONLY a JSON object with fields: "
            f"instrumentId, asOf, recommendation (BUY|HOLD|SELL), target_price (number), position_size (0-1 ensemble portfolio weight), stop_loss (multiplier vs entry), take_profit (multiplier vs entry), "
            f"horizon_days (int), rationale (string), key_risks (array of strings), catalysts (array of strings), market_context (object with optional 'sentiment': {{label, score, window_days, headlines[], method}}). "
            f"Ticker: {ticker}. As-of: {as_of}. Target language: {language}."
        )
        try:
            invoke_cfg = {"callbacks": callbacks} if callbacks else None
            # Use synchronous invoke to avoid creating/closing event loops which may cause 'Event loop is closed'
            result = await agent.ainvoke({"messages": user_msg}, config=invoke_cfg)
            # Extract assistant text
            messages = result.get("messages") if isinstance(result, dict) else None
            content = None
            if isinstance(messages, list) and messages:
                # take the last assistant content
                last = messages[-1]
                content = getattr(last, "content", None) or (last.get("content") if isinstance(last, dict) else None)
            if not content and isinstance(result, dict):
                content = result.get("final", None)
            if isinstance(content, list):
                content = "".join(x if isinstance(x, str) else str(x) for x in content)
            if not isinstance(content, str):
                raise ValueError("No text content from agent")
            # Find JSON in content
            m = re.search(r"\{[\s\S]*\}", content)
            if not m:
                raise ValueError("No JSON found in agent output")
            data = json.loads(m.group(0))
            from .schemas import InvestmentAdvice as AdviceSchema
            return AdviceSchema(**data)
        except Exception as e:
            raise RuntimeError(f"Failed to generate advice with tools: {e}")
    # Fallback: no tools, use structured output LLM
    try:
        from langchain_core.prompts import ChatPromptTemplate  # type: ignore
        from .schemas import InvestmentAdvice as AdviceSchema
    except Exception as e:
        raise RuntimeError("LangChain is required") from e
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a disciplined buy-side PM. Output a structured InvestmentAdvice. "
            "Be conservative on sizing based on volatility/liquidity and ensure stop-loss/take-profit are coherent with target and horizon. Use the target language: {language}",
        ),
        (
            "human",
            "Ticker: {ticker}\nAs-of: {as_of}\nPlease return a complete advice with BUY/HOLD/SELL, target_price, position_size (0-1), stop_loss, take_profit, horizon_days, rationale, key_risks, catalysts, market_context (echo key signals).",
        ),
    ])
    chain = prompt | llm.with_structured_output(AdviceSchema, method="function_calling")
    return chain.invoke({"ticker": ticker, "as_of": as_of, "language": language})
