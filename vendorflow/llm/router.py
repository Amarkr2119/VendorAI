"""LLM router: primary -> fallback -> (caller falls back to deterministic extraction).

Behaviour by failure kind:
  QUOTA      cool the provider down (Retry-After if given) and move straight to the next provider
  TRANSIENT  retry with exponential backoff, then move on; 3 consecutive failures open a cooldown (circuit breaker)
  AUTH/CONFIG disable the provider for the session
  BAD_OUTPUT one repair retry on the same provider, then move on
Every attempt is audited (provider, status, latency, attempt number, which provider it fell back from).
The LLM never makes business decisions: it only structures document text.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, TypeVar

from vendorflow.config import Settings
from vendorflow.llm.base import LlmClient, LlmError, LlmErrorKind, LlmUnavailable
from vendorflow.logging_setup import get_logger

log = get_logger("llm")
T = TypeVar("T")
CIRCUIT_THRESHOLD = 3


@dataclass
class _State:
    cooldown_until: float = 0.0
    disabled_reason: Optional[str] = None
    consecutive_failures: int = 0
    last_call: float = -1e9


@dataclass
class RouterResult:
    parsed: Any
    provider: str
    model: str
    attempts: int
    fallback_from: Optional[str]


class LlmRouter:
    def __init__(self, clients: Sequence[LlmClient], settings: Settings, audit: Optional[Callable[..., None]] = None,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        self.clients = list(clients)
        self.settings = settings
        self.audit = audit or (lambda **kw: None)
        self.clock, self.sleep = clock, sleep
        self.state: Dict[str, _State] = {c.name: _State() for c in self.clients}

    # ------------------------------------------------------------------ introspection
    def providers(self) -> List[str]:
        return [c.name for c in self.clients]

    def has_providers(self) -> bool:
        return bool(self.clients)

    def status(self) -> Dict[str, str]:
        out = {}
        for c in self.clients:
            st = self.state[c.name]
            out[c.name] = (f"disabled: {st.disabled_reason}" if st.disabled_reason
                           else "cooling down" if self.clock() < st.cooldown_until else "ready")
        return out

    # ------------------------------------------------------------------ main entry
    def complete_json(self, system: str, user: str, validator: Callable[[str], T], *, run_id: Optional[int] = None,
                      purpose: str = "extraction") -> RouterResult:
        attempts_log: List[Dict[str, Any]] = []
        prev_failed: Optional[str] = None
        total_attempts = 0
        for client in self.clients:
            st = self.state[client.name]
            if st.disabled_reason or self.clock() < st.cooldown_until:
                why = st.disabled_reason or "cooling down"
                self._audit(run_id, client, "SKIPPED", purpose, error=why, prev=prev_failed, attempt=0)
                attempts_log.append({"provider": client.name, "status": "SKIPPED", "error": why})
                prev_failed = client.name
                continue
            max_attempts = 1 + max(0, self.settings.llm_max_retries)
            n, repaired, user_text = 0, False, user
            while n < max_attempts:
                n += 1
                total_attempts += 1
                self._space(st)
                t0 = self.clock()
                try:
                    resp = client.generate_json(system, user_text, self.settings.llm_timeout_s)
                except LlmError as e:
                    ms = int((self.clock() - t0) * 1000)
                    self._audit(run_id, client, e.kind.value, purpose, error=e.message, ms=ms, prev=prev_failed, attempt=n, size=len(user_text))
                    attempts_log.append({"provider": client.name, "status": e.kind.value, "error": e.message})
                    if e.kind == LlmErrorKind.QUOTA:
                        st.cooldown_until = self.clock() + (e.retry_after or self.settings.llm_quota_cooldown_s)
                        break
                    if e.kind in (LlmErrorKind.AUTH, LlmErrorKind.CONFIG):
                        st.disabled_reason = f"{e.kind.value}: {e.message[:120]}"
                        log.warning("LLM provider %s disabled for this session (%s)", client.name, e.kind.value)
                        break
                    if e.kind == LlmErrorKind.TRANSIENT:
                        st.consecutive_failures += 1
                        if n < max_attempts:
                            self.sleep(min(2 ** (n - 1), 8))
                            continue
                    break
                ms = resp.latency_ms or int((self.clock() - t0) * 1000)
                try:
                    parsed = validator(resp.text)
                except ValueError as ve:
                    self._audit(run_id, client, "BAD_OUTPUT", purpose, error=str(ve)[:200], ms=ms, prev=prev_failed, attempt=n, size=len(user_text))
                    attempts_log.append({"provider": client.name, "status": "BAD_OUTPUT", "error": str(ve)[:200]})
                    if not repaired and n < max_attempts + 1:
                        repaired = True
                        max_attempts = max(max_attempts, n + 1)
                        user_text = (user + "\n\nYour previous reply was not usable (" + str(ve)[:160] +
                                     "). Reply again with ONLY the JSON object described above.")
                        continue
                    break
                st.consecutive_failures = 0
                self._audit(run_id, client, "OK", purpose, ms=ms, prev=prev_failed, attempt=n, size=len(user_text))
                return RouterResult(parsed, client.name, client.model, total_attempts, prev_failed)
            if st.consecutive_failures >= CIRCUIT_THRESHOLD:
                st.cooldown_until = self.clock() + self.settings.llm_quota_cooldown_s
                log.warning("LLM provider %s circuit opened after %d consecutive failures", client.name, st.consecutive_failures)
            prev_failed = client.name
        raise LlmUnavailable(attempts_log)

    # ------------------------------------------------------------------ helpers
    def _space(self, st: _State) -> None:
        wait = self.settings.llm_min_interval_s - (self.clock() - st.last_call)
        if wait > 0:
            self.sleep(wait)
        st.last_call = self.clock()

    def _audit(self, run_id, client, status, purpose, error=None, ms=None, prev=None, attempt=None, size=None):
        self.audit(run_id=run_id, category="LLM", provider=client.name, status=status, endpoint=getattr(client, "model", ""),
                   purpose=purpose, request={"model": getattr(client, "model", ""), "prompt_chars": size},
                   latency_ms=ms, attempt=attempt, fallback_from=prev, error=error)


def build_router(settings: Settings, audit: Optional[Callable[..., None]] = None, session: Any = None, **kw) -> LlmRouter:
    """Create clients for the configured primary/fallback providers that have BOTH a key and a model set."""
    from vendorflow.llm.gemini_client import GeminiClient
    from vendorflow.llm.groq_client import GroqClient

    clients: List[LlmClient] = []
    if settings.llm_enabled:
        for name in [settings.llm_primary, settings.llm_fallback]:
            if name in {c.name for c in clients} or not settings.provider_configured(name):
                continue
            if name == "gemini":
                clients.append(GeminiClient(settings.gemini_api_key, settings.gemini_model, session))
            elif name == "groq":
                clients.append(GroqClient(settings.groq_api_key, settings.groq_model, session))
    return LlmRouter(clients, settings, audit, **kw)
