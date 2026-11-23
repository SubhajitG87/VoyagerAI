"""Progress events emitted while a trip is planned."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from voyager_ai.models import AgentResult, TripPlan

EventKind = Literal[
    "agent_started", "agent_chunk", "agent_finished", "agent_failed", "plan_finished"
]


@dataclass(frozen=True)
class PlanEvent:
    """One step of a plan run.

    `text` holds the streamed text for `agent_chunk` and the error for `agent_failed`.
    """

    kind: EventKind
    agent_id: str | None = None
    text: str = ""
    result: AgentResult | None = None
    plan: TripPlan | None = None

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe summary. The full plan is left out; it is stored separately."""

        return {
            "kind": self.kind,
            "agent_id": self.agent_id,
            "text": self.text,
            "result": _result_summary(self.result) if self.result is not None else None,
        }


EventSink = Callable[[PlanEvent], None]


def _result_summary(result: AgentResult) -> dict[str, Any]:
    report = result.report
    summary: dict[str, Any] = {
        "status": result.status,
        "model": result.model,
        "error": result.error,
        "warnings": list(result.warnings),
        "findings": len(report.findings) if report else 0,
        "sourced": report.sourced_count if report else 0,
    }
    if result.itinerary is not None:
        summary["days"] = len(result.itinerary.days)
    return summary
