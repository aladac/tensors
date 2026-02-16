"""ComfyUI reverse proxy with GitHub OAuth authentication."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import secrets
import time

import httpx
import websockets
from fastapi import APIRouter, Cookie, HTTPException, Request, Response, WebSocket, status
from fastapi.responses import HTMLResponse, RedirectResponse

router = APIRouter(tags=["ComfyUI"])

# Number of parts in session token
_SESSION_TOKEN_PARTS = 3

# Config from environment
COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
SESSION_SECRET = os.environ.get("SESSION_SECRET", "tensors-comfyui-secret-change-me")
SESSION_MAX_AGE = 86400 * 7  # 7 days

# GitHub OAuth config
GITHUB_CLIENT_ID = os.environ.get("GITHUB_CLIENT_ID", "")
GITHUB_CLIENT_SECRET = os.environ.get("GITHUB_CLIENT_SECRET", "")
GITHUB_ALLOWED_USERS = os.environ.get("GITHUB_ALLOWED_USERS", "").split(",")

# OAuth state storage (in-memory, short-lived)
_oauth_states: dict[str, float] = {}


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


def _is_auth_configured() -> bool:
    """Check if GitHub OAuth is configured."""
    return bool(GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET)


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
        .github-btn {
            width: 100%;
            padding: 14px;
            font-size: 16px;
            font-weight: 600;
            border: none;
            border-radius: 8px;
            background: #24292f;
            color: #fff;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s, background 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            text-decoration: none;
        }
        .github-btn:hover {
            background: #32383f;
            transform: translateY(-2px);
            box-shadow: 0 10px 20px -10px rgba(0, 0, 0, 0.5);
        }
        .github-btn:active {
            transform: translateY(0);
        }
        .github-btn svg {
            width: 20px;
            height: 20px;
            fill: currentColor;
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
        <a href="/comfy/auth/github" class="github-btn">
            <svg viewBox="0 0 16 16" aria-hidden="true">
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
                    0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13
                    -.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66
                    .07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15
                    -.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0
                    1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82
                    1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01
                    1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
            </svg>
            Sign in with GitHub
        </a>
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


@router.get("/comfy/auth/github")
async def github_auth(request: Request) -> Response:
    """Redirect to GitHub OAuth."""
    if not _is_auth_configured():
        return RedirectResponse(
            url="/comfy/login?error=GitHub+OAuth+not+configured",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    # Generate state for CSRF protection
    state = secrets.token_urlsafe(32)
    _oauth_states[state] = time.time()

    # Clean up old states (older than 10 minutes)
    cutoff = time.time() - 600
    for s in list(_oauth_states.keys()):
        if _oauth_states[s] < cutoff:
            del _oauth_states[s]

    # Build GitHub OAuth URL
    params = {
        "client_id": GITHUB_CLIENT_ID,
        "redirect_uri": str(request.url_for("github_callback")),
        "scope": "read:user",
        "state": state,
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return RedirectResponse(
        url=f"https://github.com/login/oauth/authorize?{query}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/comfy/auth/callback")
async def github_callback(  # noqa: PLR0911
    code: str | None = None,
    state: str | None = None,
) -> Response:
    """Handle GitHub OAuth callback."""
    # Verify state
    if not state or state not in _oauth_states:
        return RedirectResponse(
            url="/comfy/login?error=Invalid+OAuth+state",
            status_code=status.HTTP_303_SEE_OTHER,
        )
    del _oauth_states[state]

    if not code:
        return RedirectResponse(
            url="/comfy/login?error=No+authorization+code",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    # Exchange code for access token
    async with httpx.AsyncClient() as client:
        token_response = await client.post(
            "https://github.com/login/oauth/access_token",
            data={
                "client_id": GITHUB_CLIENT_ID,
                "client_secret": GITHUB_CLIENT_SECRET,
                "code": code,
            },
            headers={"Accept": "application/json"},
        )
        token_data = token_response.json()

    if "error" in token_data:
        return RedirectResponse(
            url=f"/comfy/login?error={token_data.get('error_description', 'OAuth+error')}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    access_token = token_data.get("access_token")
    if not access_token:
        return RedirectResponse(
            url="/comfy/login?error=No+access+token",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    # Get user info
    async with httpx.AsyncClient() as client:
        user_response = await client.get(
            "https://api.github.com/user",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/vnd.github+json",
            },
        )
        user_data = user_response.json()

    username = user_data.get("login", "")
    if not username:
        return RedirectResponse(
            url="/comfy/login?error=Could+not+get+GitHub+username",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    # Check if user is allowed
    allowed = [u.strip().lower() for u in GITHUB_ALLOWED_USERS if u.strip()]
    if allowed and username.lower() not in allowed:
        return RedirectResponse(
            url="/comfy/login?error=User+not+authorized",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    # Create session
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
    if not _is_auth_configured():
        return

    if method == "OPTIONS":
        return

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

    # Forward headers (excluding problematic ones)
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("cookie", None)
    headers.pop("origin", None)  # ComfyUI blocks requests with Origin header

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
    if _is_auth_configured() and not _verify_session_token(comfy_session):
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
