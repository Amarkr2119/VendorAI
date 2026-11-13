"""India adapters. No public government API exists for GSTIN/PAN/bank-account verification: those need a licensed
provider (GSP / NSDL-Protean / penny-drop vendors). Until one is plugged in via ProviderRegistry.register(), the only
implementations here are SIMULATED. IFSC: format is checked locally; no authoritative lookup source is bundled."""
from __future__ import annotations

from typing import Any, Dict, Optional

from vendorflow.verification.simulated import SimulatedAdapter


def simulated_adapters(unknown: str = "accept", fixtures: Optional[Dict[str, Any]] = None):
    mk = lambda ct, sec, nm, nk="name": SimulatedAdapter("IN", ct, sec, nm, name_key=nk, unknown=unknown, fixtures=fixtures)
    return [mk("GSTIN_VERIFY", "gstin", "sim-in-gstin", "legal_name"),
            mk("PAN_VERIFY", "pan", "sim-in-pan"),
            mk("BANK_VERIFY", "bank", "sim-in-bank", "holder_name")]
