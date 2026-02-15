"""Pytest configuration and fixtures."""

from __future__ import annotations

import base64
import json
import struct

import pytest

# 1x1 red PNG for image response stubs
TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
    b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)
TINY_PNG_B64 = base64.b64encode(TINY_PNG).decode()


@pytest.fixture
def temp_safetensor(tmp_path):
    """Create a minimal valid safetensor file for testing."""

    # Create minimal safetensor with empty tensors and some metadata
    header = {
        "__metadata__": {
            "format": "pt",
            "test_key": "test_value",
        }
    }
    header_bytes = json.dumps(header).encode("utf-8")
    header_size = len(header_bytes)

    file_path = tmp_path / "test_model.safetensors"
    with file_path.open("wb") as f:
        f.write(struct.pack("<Q", header_size))
        f.write(header_bytes)

    return file_path
