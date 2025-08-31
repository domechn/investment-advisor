from __future__ import annotations

import argparse
import logging
import os
from dotenv import load_dotenv

from ai_advisor.orchestrator import run_pipeline
from ai_advisor.storage import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the investment advisor agentic pipeline")
    parser.add_argument("--idea", required=True, help="Idea description or ticker/keyword")
    parser.add_argument("--instrument", required=False, help="Optional instrumentId/ticker override")
    parser.add_argument("--out", default="knowledge", help="Root directory for artifacts")
    parser.add_argument("--use-mcp", action="store_true", help="Enrich with market data via MCP financial-datasets server")
    parser.add_argument("--advice", action="store_true", help="Also generate a high-level investment advice snapshot")
    parser.add_argument("--report", action="store_true", help="Render a Markdown research report")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging output")
    parser.add_argument("--lang", default="en", help="Target language for LLM outputs (e.g., en, zh). Default: en")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    # Logging setup
    logging.basicConfig(level=(logging.DEBUG if args.verbose else logging.INFO), format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    # Enforce API key
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY (set env or .env)")

    async def _amain():
        return await run_pipeline(
            idea=args.idea,
            instrument_id=args.instrument,
            root=args.out,
            use_mcp=args.use_mcp,
            generate_advice=args.advice,
            generate_report_md=args.report,
            language=args.lang,
        )
    try:
        import asyncio
        case = asyncio.run(_amain())
    except KeyboardInterrupt:
        # Exit code 130 for SIGINT
        raise SystemExit(130)
    # also write a combined one-shot report for convenience
    report_path = os.path.join(args.out, "proposals", case.instrumentId, f"report_{case.asOf}.json")
    write_json(report_path, case.model_dump())
    print(f"Generated proposal for {case.instrumentId} at {case.asOf}. Report: {report_path}")
    if args.advice:
        advice_path = os.path.join(args.out, "proposals", case.instrumentId, f"advice_{case.asOf}.json")
        print(f"Advice artifact: {advice_path}")
    if args.report:
        md_path = os.path.join(args.out, "proposals", case.instrumentId, f"report_{case.asOf}.md")
        print(f"Report: {md_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
