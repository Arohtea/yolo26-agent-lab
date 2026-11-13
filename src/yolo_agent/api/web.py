"""同源工作台与白名单说明文件，不开放项目目录浏览。"""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

WEB_ROOT = Path(__file__).resolve().parents[1] / "web"
DOCS_ROOT = Path("docs").resolve()
GUIDES = {name: f"{name}.md" for name in (
    "README", "datasets", "annotations", "training", "evaluation", "agents", "deployment", "web", "stagnation",
)}
LICENSES = {"alpinejs": "alpinejs.LICENSE.txt", "chart.js": "chart.js.LICENSE.txt", "plus-jakarta-sans": "fontsource-variable__plus-jakarta-sans.LICENSE.txt", "inter": "fontsource-variable__inter.LICENSE.txt", "jetbrains-mono": "fontsource-variable__jetbrains-mono.LICENSE.txt"}
router = APIRouter(include_in_schema=False)
assets = StaticFiles(directory=WEB_ROOT)


@router.get("/")
@router.get("/app")
def workbench() -> FileResponse:
    """返回工作台入口，资源和请求使用同一个服务来源。

    :returns: 每次重新校验的 HTML，避免升级后仍引用旧页面。
    """
    return FileResponse(WEB_ROOT / "index.html", headers={"Cache-Control": "no-store"})


@router.get("/guide/{topic}")
def guide(topic: str) -> FileResponse:
    """只读固定用户指南，不能通过 URL 指定任意本地文件。

    :param topic: 固定文档主题。
    :returns: UTF-8 Markdown 原文。
    """
    filename = GUIDES.get(topic)
    if filename is None or not (DOCS_ROOT / filename).is_file():
        raise HTTPException(404, "说明文件不存在")
    return FileResponse(DOCS_ROOT / filename, media_type="text/plain; charset=utf-8", headers={"X-Content-Type-Options": "nosniff"})


@router.get("/licenses", response_class=PlainTextResponse)
def licenses() -> str:
    """公开随工作台分发的组件许可证及固定原文链接。

    :returns: 组件版本和许可证索引。
    """
    return "YOLO26 Agent Lab Web 开源组件\n\nAlpine.js 3.15.8 · MIT\n/licenses/alpinejs\n\nChart.js 4.5.1 · MIT\n/licenses/chart.js\n\nPlus Jakarta Sans（Fontsource 5.3.0）· OFL-1.1\n/licenses/plus-jakarta-sans\n\nInter（Fontsource 5.3.0）· OFL-1.1\n/licenses/inter\n\nJetBrains Mono（Fontsource 5.3.0）· OFL-1.1\n/licenses/jetbrains-mono\n\n完整资源来源与指纹：\n/licenses/manifest\n"


@router.get("/licenses/{package}")
def license_file(package: str) -> FileResponse:
    """按固定组件名提供许可证，不接受文件路径。

    :param package: 固定组件或 manifest。
    :returns: 本地许可证原文或资源清单。
    """
    filename = "manifest.json" if package == "manifest" else LICENSES.get(package)
    if filename is None or not (DOCS_ROOT / "vendor" / filename).is_file():
        raise HTTPException(404, "许可证不存在")
    return FileResponse(DOCS_ROOT / "vendor" / filename, media_type="application/json" if package == "manifest" else "text/plain; charset=utf-8")
