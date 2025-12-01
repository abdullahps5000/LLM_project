from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Capabilities:
    name: str
    agent_id: str
    cpu_count: int
    ram_total_bytes: int
    ram_avail_bytes: int
    eff_gflops: float

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Capabilities":
        return Capabilities(
            name=str(d["name"]),
            agent_id=str(d["agent_id"]),
            cpu_count=int(d.get("cpu_count", 1)),
            ram_total_bytes=int(d.get("ram_total_bytes", 0)),
            ram_avail_bytes=int(d.get("ram_avail_bytes", 0)),
            eff_gflops=float(d.get("eff_gflops", 1.0)),
        )
