from __future__ import annotations

import http.server
import os
import socketserver
import threading
from dataclasses import dataclass
from typing import Optional


@dataclass
class ServerHandle:
    host: str
    port: int
    directory: str
    _thread: threading.Thread
    _httpd: socketserver.TCPServer

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    def join(self, timeout: Optional[float] = None) -> None:
        self._thread.join(timeout=timeout)


def serve_directory(directory: str, host: str = "0.0.0.0", port: int = 8090) -> ServerHandle:
    directory = os.path.abspath(directory)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)

        # quiet logs a bit
        def log_message(self, format: str, *args) -> None:  # type: ignore[override]
            return

    httpd = socketserver.TCPServer((host, port), Handler, bind_and_activate=True)
    # Use non-daemon thread so server stays alive during downloads
    t = threading.Thread(target=httpd.serve_forever, daemon=False)
    t.start()
    
    # Wait a moment to ensure server started
    import time
    time.sleep(0.5)
    
    return ServerHandle(host=host, port=port, directory=directory, _thread=t, _httpd=httpd)
