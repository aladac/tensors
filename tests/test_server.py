"""Tests for tensors.server package (FastAPI sd-server manager)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from tensors.server import create_app
from tensors.server.process import ProcessManager


@pytest.fixture()
def pm() -> ProcessManager:
    return ProcessManager()


@pytest.fixture()
def api() -> TestClient:
    return TestClient(create_app())


def _get_pm(api: TestClient) -> ProcessManager:
    return api.app.state.pm  # type: ignore[union-attr]


class TestStatus:
    def test_not_running(self, api: TestClient) -> None:
        r = api.get("/status")
        assert r.status_code == 200
        assert r.json()["running"] is False

    def test_running(self, api: TestClient) -> None:
        pm = _get_pm(api)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 999
        pm.proc = mock_proc
        pm.config = {"model": "/m.safetensors", "port": 1234, "args": []}
        r = api.get("/status")
        data = r.json()
        assert data["running"] is True
        assert data["pid"] == 999

    def test_exited(self, api: TestClient) -> None:
        pm = _get_pm(api)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        pm.proc = mock_proc
        r = api.get("/status")
        data = r.json()
        assert data["running"] is False
        assert data["exit_code"] == 1


class TestStart:
    @patch("tensors.server.process.subprocess.Popen")
    def test_start_success(self, mock_popen: MagicMock, api: TestClient) -> None:
        mock_popen.return_value.pid = 42
        mock_popen.return_value.poll.return_value = None
        r = api.post("/start", json={"model": "/m.safetensors"})
        assert r.status_code == 200
        assert r.json()["started"] is True
        assert r.json()["pid"] == 42

    @patch("tensors.server.process.subprocess.Popen")
    def test_start_already_running(self, mock_popen: MagicMock, api: TestClient) -> None:
        pm = _get_pm(api)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        pm.proc = mock_proc
        r = api.post("/start", json={"model": "/m.safetensors"})
        assert r.status_code == 409


class TestStop:
    def test_stop_not_running(self, api: TestClient) -> None:
        r = api.post("/stop")
        assert r.status_code == 409

    def test_stop_running(self, api: TestClient) -> None:
        pm = _get_pm(api)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.wait.return_value = 0
        pm.proc = mock_proc
        r = api.post("/stop")
        assert r.status_code == 200
        assert r.json()["stopped"] is True
        mock_proc.send_signal.assert_called_once()


class TestRestart:
    def test_restart_no_config_no_model(self, api: TestClient) -> None:
        r = api.post("/restart", json={})
        assert r.status_code == 400

    @patch("tensors.server.process.subprocess.Popen")
    def test_restart_with_new_model(self, mock_popen: MagicMock, api: TestClient) -> None:
        mock_popen.return_value.pid = 100
        mock_popen.return_value.poll.return_value = None
        pm = _get_pm(api)
        pm.config = {"model": "/old.safetensors", "port": 1234, "args": []}
        r = api.post("/restart", json={"model": "/new.safetensors"})
        assert r.status_code == 200
        data = r.json()
        assert data["restarted"] is True
        assert "/new.safetensors" in str(data["cmd"])

    @patch("tensors.server.process.subprocess.Popen")
    def test_restart_keeps_previous_config(self, mock_popen: MagicMock, api: TestClient) -> None:
        mock_popen.return_value.pid = 101
        mock_popen.return_value.poll.return_value = None
        pm = _get_pm(api)
        pm.config = {"model": "/m.safetensors", "port": 5555, "args": ["--fa"]}
        r = api.post("/restart", json={})
        assert r.status_code == 200
        assert "5555" in str(r.json()["cmd"])


class TestProcessManager:
    def test_status_not_running(self, pm: ProcessManager) -> None:
        assert pm.status() == {"running": False}

    def test_build_cmd(self, pm: ProcessManager) -> None:
        config = {"model": "/m.gguf", "port": 1234, "args": ["--fa"]}
        cmd = pm.build_cmd(config)
        assert "/m.gguf" in cmd
        assert "--fa" in cmd
        assert "1234" in cmd

    @patch("tensors.server.process.subprocess.Popen")
    def test_start_and_stop(self, mock_popen: MagicMock, pm: ProcessManager) -> None:
        mock_popen.return_value.pid = 77
        mock_popen.return_value.poll.return_value = None
        mock_popen.return_value.wait.return_value = 0
        pm.start({"model": "/m.gguf", "port": 1234, "args": []})
        assert pm.proc is not None
        assert pm.stop() is True
        assert pm.proc is None
