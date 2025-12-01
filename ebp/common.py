from __future__ import annotations

import hashlib
import os
import socket
from dataclasses import dataclass
from typing import Iterable


def sha256_file(path: str, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def human_bytes(n: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(n)
    for u in units:
        if v < 1024.0:
            return f"{v:.2f}{u}"
        v /= 1024.0
    return f"{v:.2f}PB"


def pick_bind_ip_for_peer(peer_ip: str = "8.8.8.8") -> str:
    """
    Returns the local IP address that would be used to reach peer_ip.
    Useful for telling other devices where the PC's stage server lives.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((peer_ip, 80))
        return s.getsockname()[0]
    finally:
        s.close()


@dataclass(frozen=True)
class StageRef:
    stage_id: str
    url: str
    bytes: int
    sha256: str
