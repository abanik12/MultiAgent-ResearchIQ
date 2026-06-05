#!/usr/bin/env python3
"""CLI to run the ResearchIQ coordinator and full Phase 1 graph."""

import asyncio
import json
import sys
from pathlib import Path

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.agents.coordinator import plan_research
from src.graph.graph import run_research


async def main() -> None:
    load_dotenv()

    if len(sys.argv) < 2:
        print('Usage: python scripts/run_coordinator.py "your research query"')
        print('       python scripts/run_coordinator.py --graph "your research query"')
        sys.exit(1)

    use_graph = sys.argv[1] == "--graph"
    query = sys.argv[2] if use_graph else sys.argv[1]

    if use_graph:
        print(f"Running full graph for: {query}\n")
        result = await run_research(query)
        print("Sub-tasks:")
        for i, task in enumerate(result["sub_tasks"], 1):
            print(f"  {i}. {task}")
        print(f"\nSynthesis: {result['synthesis']}")
    else:
        print(f"Planning research for: {query}\n")
        plan = await plan_research(query)
        print(json.dumps(plan.model_dump(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
