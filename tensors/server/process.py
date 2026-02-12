"""sd-server process lifecycle management."""

from __future__ import annotations

import logging
import shutil
import signal
import subprocess
from typing import Any

logger = logging.getLogger(__name__)

SD_SERVER_BIN = shutil.which("sd-server") or "sd-server"


class ProcessManager:
    def __init__(self) -> None:
        self.proc: subprocess.Popen[bytes] | None = None
        self.config: dict[str, Any] = {}

    def build_cmd(self, config: dict[str, Any] | None = None) -> list[str]:
        cfg = config or self.config
        cmd = [SD_SERVER_BIN, "-m", cfg["model"], "--listen-port", str(cfg["port"])]
        cmd.extend(cfg.get("args", []))
        return cmd

    def start(self, config: dict[str, Any]) -> None:
        if self.proc is not None and self.proc.poll() is None:
            raise RuntimeError("Server already running — stop it first")
        self.config = config
        cmd = self.build_cmd(config)
        self.proc = subprocess.Popen(cmd)
        logger.info("started sd-server pid=%d cmd=%s", self.proc.pid, cmd)

    def stop(self) -> bool:
        if self.proc is None or self.proc.poll() is not None:
            self.proc = None
            return False
        self.proc.send_signal(signal.SIGTERM)
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)
        logger.info("stopped sd-server")
        self.proc = None
        return True

    def status(self) -> dict[str, Any]:
        if self.proc is None:
            return {"running": False}
        rc = self.proc.poll()
        if rc is not None:
            return {"running": False, "exit_code": rc}
        return {
            "running": True,
            "pid": self.proc.pid,
            "config": self.config,
            "cmd": self.build_cmd(),
        }
