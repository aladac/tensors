"""Tests for tensors.generate package."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx
import pytest
import respx

from tensors.generate import SDClient
from tensors.generate._http import HttpTransport
from tensors.generate.params import Img2ImgParams, Txt2ImgParams
from tensors.generate.util import save_images, to_b64
from tests.conftest import BASE_URL, TINY_PNG, TINY_PNG_B64

# ── util ──────────────────────────────────────────────────────────────


class TestToB64:
    def test_bytes_input(self):
        raw = b"hello"
        assert to_b64(raw) == base64.b64encode(raw).decode()

    def test_file_path(self, tmp_path: Path):
        f = tmp_path / "img.png"
        f.write_bytes(b"\x89PNG")
        result = to_b64(str(f))
        assert base64.b64decode(result) == b"\x89PNG"

    def test_pathlib_path(self, tmp_path: Path):
        f = tmp_path / "img.png"
        f.write_bytes(b"data")
        result = to_b64(f)
        assert base64.b64decode(result) == b"data"

    def test_passthrough_string(self):
        b64 = base64.b64encode(b"already").decode()
        assert to_b64(b64) == b64

    def test_unsupported_type(self):
        with pytest.raises(TypeError, match="unsupported image type"):
            to_b64(12345)  # type: ignore[arg-type]


class TestSaveImages:
    def test_saves_files(self, tmp_path: Path):
        images = [b"img0", b"img1", b"img2"]
        paths = save_images(images, str(tmp_path), prefix="test")
        assert len(paths) == 3
        for i, p in enumerate(paths):
            assert p.name == f"test_{i:04d}.png"
            assert p.read_bytes() == images[i]

    def test_creates_directory(self, tmp_path: Path):
        out = tmp_path / "sub" / "dir"
        save_images([b"x"], str(out))
        assert (out / "output_0000.png").exists()


# ── params ────────────────────────────────────────────────────────────


class TestTxt2ImgParams:
    def test_minimal_body(self):
        p = Txt2ImgParams(prompt="a cat")
        body = p.to_body()
        assert body["prompt"] == "a cat"
        assert body["width"] == 512
        assert body["height"] == 512
        assert body["steps"] == 20
        assert body["seed"] == -1
        assert "sampler_name" not in body
        assert "scheduler" not in body
        assert "clip_skip" not in body
        assert "lora" not in body

    def test_optional_fields_included(self):
        p = Txt2ImgParams(
            prompt="test",
            sampler_name="euler_a",
            scheduler="karras",
            clip_skip=2,
            lora=[{"path": "x.safetensors", "multiplier": 0.5}],
        )
        body = p.to_body()
        assert body["sampler_name"] == "euler_a"
        assert body["scheduler"] == "karras"
        assert body["clip_skip"] == 2
        assert len(body["lora"]) == 1


class TestImg2ImgParams:
    def test_minimal_body(self, tmp_path: Path):
        img = tmp_path / "init.png"
        img.write_bytes(b"\x89PNG")
        p = Img2ImgParams(prompt="paint it", init_image=str(img))
        body = p.to_body()
        assert body["prompt"] == "paint it"
        assert body["denoising_strength"] == 0.75
        decoded = base64.b64decode(body["init_images"][0])
        assert decoded == b"\x89PNG"
        assert "width" not in body
        assert "height" not in body
        assert "mask" not in body

    def test_all_optional_fields(self, tmp_path: Path):
        img = tmp_path / "init.png"
        img.write_bytes(b"img")
        mask = tmp_path / "mask.png"
        mask.write_bytes(b"mask")
        extra = tmp_path / "extra.png"
        extra.write_bytes(b"extra")

        p = Img2ImgParams(
            prompt="test",
            init_image=str(img),
            mask=str(mask),
            width=768,
            height=768,
            inpainting_mask_invert=True,
            sampler_name="euler",
            scheduler="simple",
            clip_skip=1,
            lora=[{"path": "a.gguf", "multiplier": 1.0}],
            extra_images=[str(extra)],
        )
        body = p.to_body()
        assert body["width"] == 768
        assert body["mask"]
        assert body["inpainting_mask_invert"] == 1
        assert body["sampler_name"] == "euler"
        assert len(body["extra_images"]) == 1


# ── _http ─────────────────────────────────────────────────────────────


class TestHttpTransport:
    def test_get_success(self):
        with respx.mock(base_url=BASE_URL) as rsps:
            rsps.get("/test").respond(json={"ok": True})
            t = HttpTransport(BASE_URL)
            assert t.get("/test") == {"ok": True}
            t.close()

    def test_post_success(self):
        with respx.mock(base_url=BASE_URL) as rsps:
            rsps.post("/gen").respond(json={"images": []})
            t = HttpTransport(BASE_URL)
            assert t.post("/gen", {"prompt": "x"}) == {"images": []}
            t.close()

    def test_get_http_error(self):
        with respx.mock(base_url=BASE_URL) as rsps:
            rsps.get("/bad").respond(status_code=404, text="not found")
            t = HttpTransport(BASE_URL)
            with pytest.raises(httpx.HTTPStatusError):
                t.get("/bad")
            t.close()

    def test_post_http_error(self):
        with respx.mock(base_url=BASE_URL) as rsps:
            rsps.post("/bad").respond(status_code=500, text="error")
            t = HttpTransport(BASE_URL)
            with pytest.raises(httpx.HTTPStatusError):
                t.post("/bad", {})
            t.close()

    def test_get_connection_error(self):
        with respx.mock(base_url=BASE_URL) as rsps:
            rsps.get("/fail").mock(side_effect=httpx.ConnectError("refused"))
            t = HttpTransport(BASE_URL)
            with pytest.raises(httpx.ConnectError):
                t.get("/fail")
            t.close()


# ── info ──────────────────────────────────────────────────────────────


class TestInfoAPI:
    def test_models(self, mock_api: respx.MockRouter, client: SDClient):
        mock_api.get("/v1/models").respond(json={"data": [{"id": "sd-cpp-local", "object": "model", "owned_by": "local"}]})
        result = client.info.models()
        assert len(result) == 1
        assert result[0]["id"] == "sd-cpp-local"

    def test_sd_models(self, mock_api: respx.MockRouter, client: SDClient):
        mock_api.get("/sdapi/v1/sd-models").respond(
            json=[{"title": "sdxl", "model_name": "sdxl", "filename": "sdxl.safetensors"}]
        )
        result = client.info.sd_models()
        assert result[0]["title"] == "sdxl"

    def test_options(self, mock_api: respx.MockRouter, client: SDClient):
        mock_api.get("/sdapi/v1/options").respond(
            json={
                "samples_format": "png",
                "sd_model_checkpoint": "v1-5",
            }
        )
        result = client.info.options()
        assert result["sd_model_checkpoint"] == "v1-5"

    def test_loras(self, mock_api: respx.MockRouter, client: SDClient):
        mock_api.get("/sdapi/v1/loras").respond(
            json=[
                {"name": "style", "path": "style.safetensors"},
            ]
        )
        result = client.info.loras()
        assert len(result) == 1
        assert result[0]["name"] == "style"

    def test_samplers(self, mock_api: respx.MockRouter, client: SDClient):
        mock_api.get("/sdapi/v1/samplers").respond(
            json=[
                {"name": "euler", "aliases": ["euler"], "options": {}},
                {"name": "euler_a", "aliases": ["euler_a"], "options": {}},
            ]
        )
        result = client.info.samplers()
        assert result == ["euler", "euler_a"]

    def test_schedulers(self, mock_api: respx.MockRouter, client: SDClient):
        mock_api.get("/sdapi/v1/schedulers").respond(
            json=[
                {"name": "discrete", "label": "discrete"},
                {"name": "karras", "label": "karras"},
            ]
        )
        result = client.info.schedulers()
        assert result == ["discrete", "karras"]


# ── generation ────────────────────────────────────────────────────────


class TestTxt2Img:
    def test_returns_decoded_images(self, mock_api: respx.MockRouter, client: SDClient):
        mock_api.post("/sdapi/v1/txt2img").respond(
            json={
                "images": [TINY_PNG_B64],
                "parameters": {},
                "info": "",
            }
        )
        images = client.generate.txt2img(Txt2ImgParams(prompt="a cat"))
        assert len(images) == 1
        assert images[0] == TINY_PNG

    def test_multiple_images(self, mock_api: respx.MockRouter, client: SDClient):
        mock_api.post("/sdapi/v1/txt2img").respond(
            json={
                "images": [TINY_PNG_B64, TINY_PNG_B64, TINY_PNG_B64],
                "parameters": {},
                "info": "",
            }
        )
        params = Txt2ImgParams(prompt="cats", batch_size=3)
        images = client.generate.txt2img(params)
        assert len(images) == 3

    def test_sends_correct_body(self, mock_api: respx.MockRouter, client: SDClient):
        route = mock_api.post("/sdapi/v1/txt2img").respond(
            json={
                "images": [TINY_PNG_B64],
                "parameters": {},
                "info": "",
            }
        )
        params = Txt2ImgParams(
            prompt="hello",
            width=768,
            height=768,
            steps=30,
            sampler_name="euler_a",
        )
        client.generate.txt2img(params)
        sent = json.loads(route.calls[0].request.content)
        assert sent["prompt"] == "hello"
        assert sent["width"] == 768
        assert sent["sampler_name"] == "euler_a"


class TestImg2Img:
    def test_returns_decoded_images(self, mock_api: respx.MockRouter, client: SDClient, tmp_path: Path):
        mock_api.post("/sdapi/v1/img2img").respond(
            json={
                "images": [TINY_PNG_B64],
                "parameters": {},
                "info": "",
            }
        )
        img = tmp_path / "init.png"
        img.write_bytes(TINY_PNG)
        params = Img2ImgParams(prompt="paint", init_image=str(img))
        images = client.generate.img2img(params)
        assert len(images) == 1
        assert images[0] == TINY_PNG
