#!/usr/bin/env python3
"""CLI to run the ResearchIQ coordinator and full Phase 1 graph."""

import asyncio
import json
import sys
from pathlib import Path

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.agents.coordinator import plan_research_with_usage
from src.graph.graph import run_research
from src.utils.token_cost import format_usage_summary


async def main() -> None:
    load_dotenv()

    if len(sys.argv) < 2:
        print('Usage: python scripts/run_coordinator.py "your research query"')
        print('       python scripts/run_coordinator.py --graph "your research query"')
        sys.exit(1)

    use_graph = sys.argv[1] == "--graph"
    query = sys.argv[2] if use_graph else sys.argv[1]

    if use_graph:
        print(f"Running full research pipeline for: {query}\n")
        result = await run_research(query)
        print("Sub-tasks:")
        for i, task in enumerate(result["sub_tasks"], 1):
            print(f"  {i}. {task}")
        print(f"\nWeb findings: {len(result['web_findings'])}")
        print(f"Document findings: {len(result['doc_findings'])}")
        print("\n" + "=" * 60)
        print(result["synthesis"])
        print("=" * 60)
    else:
        print(f"Planning research for: {query}\n")
        result = await plan_research_with_usage(query)
        print(json.dumps(result.plan.model_dump(), indent=2))
        print(format_usage_summary(result.usage))


if __name__ == "__main__":
    asyncio.run(main())
