"""ComfyUI reverse proxy with session authentication."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import time

import httpx
import websockets
from fastapi import APIRouter, Cookie, Form, HTTPException, Request, Response, WebSocket, status
from fastapi.responses import HTMLResponse, RedirectResponse

router = APIRouter(tags=["ComfyUI"])

# Number of parts in session token
_SESSION_TOKEN_PARTS = 3

# Config from environment
COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
COMFYUI_USER = os.environ.get("COMFYUI_USER", "")
COMFYUI_PASS = os.environ.get("COMFYUI_PASS", "")
SESSION_SECRET = os.environ.get("SESSION_SECRET", "tensors-comfyui-secret-change-me")
SESSION_MAX_AGE = 86400 * 7  # 7 days


def _create_session_token(username: str) -> str:
    """Create a signed session token."""
    expires = int(time.time()) + SESSION_MAX_AGE
    data = f"{username}:{expires}"
    signature = hmac.new(SESSION_SECRET.encode(), data.encode(), hashlib.sha256).hexdigest()[:32]
    return f"{data}:{signature}"


def _verify_session_token(token: str | None) -> bool:
    """Verify a session token."""
    if not token:
        return False
    try:
        parts = token.split(":")
        if len(parts) != _SESSION_TOKEN_PARTS:
            return False
        username, expires_str, signature = parts
        expires = int(expires_str)
        if time.time() > expires:
            return False
        data = f"{username}:{expires_str}"
        expected = hmac.new(SESSION_SECRET.encode(), data.encode(), hashlib.sha256).hexdigest()[:32]
        return hmac.compare_digest(signature, expected)
    except (ValueError, TypeError):
        return False


LOGIN_PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ComfyUI - Login</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f0f23 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #e0e0e0;
        }
        .login-container {
            background: rgba(30, 30, 46, 0.95);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 40px;
            width: 100%;
            max-width: 400px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }
        .logo {
            text-align: center;
            margin-bottom: 32px;
        }
        .logo h1 {
            font-size: 28px;
            font-weight: 600;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .logo p {
            color: #888;
            font-size: 14px;
            margin-top: 8px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            font-size: 14px;
            font-weight: 500;
            margin-bottom: 8px;
            color: #b0b0b0;
        }
        input {
            width: 100%;
            padding: 14px 16px;
            font-size: 16px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            background: rgba(0, 0, 0, 0.3);
            color: #fff;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        input:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
        }
        input::placeholder {
            color: #666;
        }
        button {
            width: 100%;
            padding: 14px;
            font-size: 16px;
            font-weight: 600;
            border: none;
            border-radius: 8px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px -10px rgba(102, 126, 234, 0.5);
        }
        button:active {
            transform: translateY(0);
        }
        .error {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: #f87171;
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 14px;
        }
        .footer {
            text-align: center;
            margin-top: 24px;
            font-size: 12px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="login-container">
        <div class="logo">
            <h1>ComfyUI</h1>
            <p>Stable Diffusion GUI</p>
        </div>
        {{ERROR}}
        <form method="POST" action="/comfy/login">
            <div class="form-group">
                <label for="username">Username</label>
                <input type="text" id="username" name="username" placeholder="Enter username" required autofocus>
            </div>
            <div class="form-group">
                <label for="password">Password</label>
                <input type="password" id="password" name="password" placeholder="Enter password" required>
            </div>
            <button type="submit">Sign In</button>
        </form>
        <div class="footer">
            Powered by tensors
        </div>
    </div>
</body>
</html>
"""


@router.get("/comfy/login")
async def login_page(error: str | None = None) -> HTMLResponse:
    """Show login page."""
    error_html = ""
    if error:
        error_html = f'<div class="error">{error}</div>'
    html = LOGIN_PAGE_HTML.replace("{{ERROR}}", error_html)
    return HTMLResponse(content=html)


@router.post("/comfy/login")
async def login_submit(username: str = Form(...), password: str = Form(...)) -> Response:
    """Handle login form submission."""
    if not COMFYUI_USER or not COMFYUI_PASS:
        return RedirectResponse(
            url="/comfy/login?error=Authentication+not+configured",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    if username == COMFYUI_USER and password == COMFYUI_PASS:
        token = _create_session_token(username)
        response = RedirectResponse(url="/comfy/", status_code=status.HTTP_303_SEE_OTHER)
        response.set_cookie(
            key="comfy_session",
            value=token,
            max_age=SESSION_MAX_AGE,
            httponly=True,
            samesite="lax",
        )
        return response

    return RedirectResponse(
        url="/comfy/login?error=Invalid+username+or+password",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/comfy/logout")
async def logout() -> Response:
    """Clear session and redirect to login."""
    response = RedirectResponse(url="/comfy/login", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie("comfy_session")
    return response


def _check_auth(comfy_session: str | None, path: str = "", method: str = "GET") -> None:
    """Check if user is authenticated, raise redirect if not.

    Static assets (JS, CSS, fonts, images) are allowed without auth
    because modulepreload/crossorigin requests don't send cookies.
    OPTIONS requests (CORS preflight) are also allowed without auth.
    """
    if not COMFYUI_USER:
        # Auth not configured, allow access
        return

    # Allow CORS preflight requests (they don't send cookies)
    if method == "OPTIONS":
        return

    # Allow static assets without auth (modulepreload doesn't send cookies)
    if path.startswith("assets/"):
        return

    if not _verify_session_token(comfy_session):
        raise HTTPException(
            status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            headers={"Location": "/comfy/login"},
        )


@router.api_route("/comfy/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy_comfyui(request: Request, path: str, comfy_session: str | None = Cookie(default=None)) -> Response:
    """Proxy all HTTP requests to ComfyUI."""
    _check_auth(comfy_session, path, request.method)

    # Build target URL
    target_url = f"{COMFYUI_URL}/{path}"
    if request.url.query:
        target_url += f"?{request.url.query}"

    # Forward headers (excluding host)
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("cookie", None)

    # Get request body
    body = await request.body()

    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body,
            )
        except httpx.ConnectError as e:
            raise HTTPException(status_code=502, detail="ComfyUI is not running") from e
        except httpx.TimeoutException as e:
            raise HTTPException(status_code=504, detail="ComfyUI request timed out") from e

    # Build response
    excluded_headers = {"content-encoding", "content-length", "transfer-encoding", "connection"}
    response_headers = {k: v for k, v in response.headers.items() if k.lower() not in excluded_headers}

    # Add CORS headers for cross-origin requests
    response_headers["Access-Control-Allow-Origin"] = "*"
    response_headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response_headers["Access-Control-Allow-Headers"] = "*"

    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=response_headers,
        media_type=response.headers.get("content-type"),
    )


@router.websocket("/comfy/ws")
async def proxy_websocket(websocket: WebSocket, comfy_session: str | None = Cookie(default=None)) -> None:
    """Proxy WebSocket connections to ComfyUI."""
    # Check auth via cookie
    if COMFYUI_USER and not _verify_session_token(comfy_session):
        await websocket.close(code=4001, reason="Unauthorized")
        return

    await websocket.accept()

    # Connect to ComfyUI WebSocket
    comfy_ws_url = COMFYUI_URL.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
    if websocket.url.query:
        comfy_ws_url += f"?{websocket.url.query}"

    try:
        async with websockets.connect(comfy_ws_url) as comfy_ws:

            async def client_to_comfy() -> None:
                try:
                    while True:
                        data = await websocket.receive_text()
                        await comfy_ws.send(data)
                except Exception:
                    pass

            async def comfy_to_client() -> None:
                try:
                    async for message in comfy_ws:
                        if isinstance(message, bytes):
                            await websocket.send_bytes(message)
                        else:
                            await websocket.send_text(message)
                except Exception:
                    pass

            # Run both directions concurrently
            await asyncio.gather(client_to_comfy(), comfy_to_client(), return_exceptions=True)

    except Exception as e:
        await websocket.close(code=1011, reason=str(e))


def create_comfyui_router() -> APIRouter:
    """Return the ComfyUI proxy router."""
    return router
