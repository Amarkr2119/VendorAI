from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from vendorflow.domain.models import StageEvent


def make_event(stage: str, status: str, message: str, **detail: Any) -> StageEvent:
    return StageEvent(stage=stage, status=status, message=message, detail=detail,
                      ts=datetime.now(timezone.utc).isoformat(timespec="milliseconds"))


def event_dict(e: StageEvent) -> Dict[str, Any]:
    return {"stage": e.stage, "status": e.status, "message": e.message, "detail": e.detail, "ts": e.ts}
