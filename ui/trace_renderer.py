"""Render curated SSE trace events in Streamlit."""

from __future__ import annotations

from dataclasses import dataclass, field


AGENT_LABELS = {
    "coordinator": "Coordinator",
    "web_researcher": "Web Researcher",
    "doc_analyst": "Doc Analyst",
    "report_writer": "Report Writer",
}


@dataclass
class TraceState:
    phase: str = "starting"
    status_label: str = "Running research pipeline..."
    sub_tasks: list[str] = field(default_factory=list)
    strategy: str = ""
    web_count: int = 0
    doc_count: int = 0
    sections: dict[str, list[str]] = field(default_factory=dict)
    done_payload: dict | None = None
    error: str | None = None

    def section_key(self, agent: str, task_id: int | None = None) -> str:
        if task_id is None:
            return agent
        return f"{agent}:{task_id}"

    def add_line(self, agent: str, line: str, task_id: int | None = None) -> None:
        key = self.section_key(agent, task_id)
        self.sections.setdefault(key, []).append(line)

    def apply_event(self, event: dict) -> None:
        event_type = event.get("type", "unknown")

        if event_type == "phase":
            self.phase = event.get("phase", self.phase)
            message = event.get("message")
            if message:
                self.status_label = message
            return

        if event_type == "status":
            self.status_label = event.get("message", self.status_label)
            return

        if event_type == "plan":
            self.sub_tasks = event.get("sub_tasks", [])
            self.strategy = event.get("strategy", "")
            lines = [event.get("message", "Research plan ready")]
            for index, task in enumerate(self.sub_tasks, start=1):
                lines.append(f"{index}. {task}")
            if self.strategy:
                lines.append(f"Strategy: {self.strategy}")
            self.sections["coordinator"] = lines
            self.status_label = event.get("message", "Research plan ready")
            return

        if event_type == "agent_start":
            agent = event.get("agent", "agent")
            task_id = event.get("task_id")
            label = AGENT_LABELS.get(agent, agent.replace("_", " ").title())
            if task_id is not None:
                label = f"{label} · task {int(task_id) + 1}"
            self.add_line(agent, event.get("message", f"{label} started"), task_id=task_id)
            self.status_label = event.get("message", self.status_label)
            return

        if event_type == "agent_end":
            agent = event.get("agent", "agent")
            task_id = event.get("task_id")
            self.add_line(agent, event.get("message", "Complete"), task_id=task_id)
            return

        if event_type == "tool_call":
            agent = event.get("agent", "web_researcher")
            task_id = event.get("task_id")
            tool = event.get("tool", "tool")
            summary = event.get("input_summary", "")
            self.add_line(agent, f"Tool: {tool}({summary})", task_id=task_id)
            return

        if event_type == "trace":
            agent = event.get("agent", "agent")
            task_id = event.get("task_id")
            self.add_line(agent, event.get("message", ""), task_id=task_id)
            return

        if event_type == "web_finding":
            self.web_count += 1
            finding = event.get("finding", {})
            title = finding.get("title") or finding.get("url") or "Web result"
            self.add_line("web_researcher", f"Source: {title}")
            return

        if event_type == "doc_finding":
            self.doc_count += 1
            finding = event.get("finding", {})
            title = finding.get("title") or finding.get("source") or "KB chunk"
            score = finding.get("score")
            if score is not None:
                self.add_line("doc_analyst", f"Chunk: {title} (score={float(score):.2f})")
            else:
                self.add_line("doc_analyst", f"Chunk: {title}")
            return

        if event_type == "export":
            self.add_line("report_writer", event.get("message", "Report exported"))
            self.status_label = event.get("message", "Report exported")
            return

        if event_type == "done":
            self.done_payload = event.get("result")
            self.status_label = "Research complete"
            return

        if event_type == "error":
            self.error = event.get("message", "Unknown error")
            self.status_label = self.error


def render_trace_markdown(state: TraceState) -> str:
    order = ["coordinator", "web_researcher", "doc_analyst", "report_writer"]
    parts: list[str] = []

    for agent in order:
        exact_lines = state.sections.get(agent)
        if exact_lines:
            parts.append(f"**{AGENT_LABELS.get(agent, agent)}**")
            parts.extend(f"- {line}" for line in exact_lines)
            parts.append("")

        prefix = f"{agent}:"
        task_keys = sorted(key for key in state.sections if key.startswith(prefix))
        for key in task_keys:
            task_id = key.split(":", 1)[1]
            parts.append(f"**{AGENT_LABELS.get(agent, agent)} · task {int(task_id) + 1}**")
            parts.extend(f"- {line}" for line in state.sections[key])
            parts.append("")

    if not parts:
        return "_Waiting for agent activity..._"
    return "\n".join(parts).strip()
