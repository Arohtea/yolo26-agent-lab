"""为同主机嵌入审核建立官方 CVAT Cookie 会话，不向前端传递账号密码或 API Token。"""

import time
from urllib.parse import urlsplit, urlunsplit

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse

from yolo_agent.services.datasets import DatasetError
from yolo_agent.settings import Settings

SESSION_COOKIES = {"sessionid", "sessionfresh", "csrftoken"}
LOOPBACK = {"localhost", "127.0.0.1", "::1"}


def embedded_task_url(settings: Settings, request: Request, task_id: int) -> str:
    """Cookie 不区分端口；只允许配置中的同主机 CVAT，不能跨域下发会话。"""
    public = urlsplit(settings.cvat_public_url or settings.cvat_url)
    host = public.hostname
    if public.scheme not in {"http", "https"} or not host or public.username or public.password:
        raise DatasetError(503, "CVAT 公共地址配置无效")
    if host in LOOPBACK and request.url.hostname in LOOPBACK:
        host = request.url.hostname
    if host != request.url.hostname:
        raise DatasetError(503, "免重复登录要求 CVAT 与工作台使用相同主机名，请统一公共访问地址")
    if request.url.scheme == "https" and public.scheme != "https":
        raise DatasetError(503, "HTTPS 工作台需要配置 HTTPS 审核地址")
    netloc = f"[{host}]" if ":" in host else host
    if public.port:
        netloc += f":{public.port}"
    return urlunsplit((public.scheme, netloc, f"{public.path.rstrip('/')}/tasks/{task_id}", "", ""))


def create_browser_session(settings: Settings, request: Request, task_id: int) -> JSONResponse:
    """复用有效浏览器会话，否则使用部署凭据登录；只下发认证所需 Cookie。"""
    target = embedded_task_url(settings, request, task_id)
    origin = request.headers.get("origin")
    if origin and origin != str(request.base_url).rstrip("/"):
        raise DatasetError(403, "只能从当前工作台建立审核会话")
    if not settings.cvat_username or not settings.cvat_password.get_secret_value():
        raise DatasetError(503, "服务端尚未配置 CVAT 操作账户")
    incoming = {name: value for name, value in request.cookies.items() if name in SESSION_COOKIES}
    try:
        with httpx.Client(base_url=settings.cvat_url.rstrip("/"), timeout=30, follow_redirects=False, cookies=incoming) as client:
            reusable = False
            if SESSION_COOKIES <= incoming.keys():
                identity = client.get("/api/users/self")
                reusable = identity.status_code == 200 and client.get(f"/api/tasks/{task_id}").status_code == 200
            if not reusable:
                client.cookies.clear()
                login = client.post("/api/auth/login", json={"username": settings.cvat_username,
                                    "password": settings.cvat_password.get_secret_value()})
                if login.status_code != 200:
                    raise DatasetError(502, "CVAT 自动登录失败，请检查服务端操作账户配置")
                if client.get("/api/users/self").status_code != 200:
                    raise DatasetError(502, "CVAT 未建立有效浏览器会话")
                accessible = client.get(f"/api/tasks/{task_id}")
                if accessible.status_code != 200:
                    raise DatasetError(502, "CVAT 审核任务暂不可访问，请稍后重新连接")
            cookies = {cookie.name: cookie for cookie in client.cookies.jar if cookie.name in SESSION_COOKIES}
            if not SESSION_COOKIES <= cookies.keys():
                raise DatasetError(502, "CVAT 返回的审核会话不完整")
            response = JSONResponse({"url": target}, headers={"Cache-Control": "no-store", "Pragma": "no-cache"})
            for name, cookie in cookies.items():
                response.set_cookie(name, cookie.value, path="/", httponly=name != "csrftoken",
                                    secure=request.url.scheme == "https", samesite="lax",
                                    max_age=max(0, int(cookie.expires - time.time())) if cookie.expires else None)
            return response
    except httpx.RequestError as error:
        raise DatasetError(503, "暂时无法连接 CVAT，请稍后点击重新连接") from error
