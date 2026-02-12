"""FastAPI route handlers for the wrapper API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException

from tensors.server.models import RestartRequest, StartRequest  # noqa: TC001

if TYPE_CHECKING:
    from tensors.server.process import ProcessManager


def create_router(pm: ProcessManager) -> APIRouter:
    """Build a new router bound to the given ProcessManager."""
    router = APIRouter()

    @router.get("/status")
    def status() -> dict[str, Any]:
        return pm.status()

    @router.post("/start")
    def start(req: StartRequest) -> dict[str, Any]:
        if pm.proc is not None and pm.proc.poll() is None:
            raise HTTPException(409, "Server already running — use /restart or /stop first")
        config = {"model": req.model, "port": req.port, "args": req.args}
        pm.start(config)
        assert pm.proc is not None
        return {"started": True, "pid": pm.proc.pid, "cmd": pm.build_cmd(config)}

    @router.post("/stop")
    def stop() -> dict[str, Any]:
        if not pm.stop():
            raise HTTPException(409, "Server is not running")
        return {"stopped": True}

    @router.post("/restart")
    def restart(req: RestartRequest) -> dict[str, Any]:
        if not pm.config and req.model is None:
            raise HTTPException(400, "No previous config — provide at least 'model'")
        config = dict(pm.config)
        if req.model is not None:
            config["model"] = req.model
        if req.port is not None:
            config["port"] = req.port
        if req.args is not None:
            config["args"] = req.args
        was_running = pm.stop()
        pm.start(config)
        assert pm.proc is not None
        return {
            "restarted": True,
            "was_running": was_running,
            "pid": pm.proc.pid,
            "cmd": pm.build_cmd(config),
        }

    return router
