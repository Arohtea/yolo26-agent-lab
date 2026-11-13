"""显式准备固定版本的成熟前端库，运行页面不依赖外部 CDN。"""

import base64
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path
from urllib.parse import urlparse

import httpx

PACKAGES=(
    {"name":"alpinejs","version":"3.15.8","member":"package/dist/cdn.min.js","target":"alpine.min.js","license_url":"https://raw.githubusercontent.com/alpinejs/alpine/v3.15.8/LICENSE.md"},
    {"name":"chart.js","version":"4.5.1","member":"package/dist/chart.umd.min.js","target":"chart.min.js","license_member":"package/LICENSE.md"},
    # 界面字体只取拉丁可变字重文件，中文回退到系统字体，避免引入数十 MB 的 CJK 字库。
    {"name":"@fontsource-variable/plus-jakarta-sans","version":"5.3.0","member":"package/files/plus-jakarta-sans-latin-wght-normal.woff2","target":"plus-jakarta-sans.woff2","license_member":"package/LICENSE"},
    {"name":"@fontsource-variable/inter","version":"5.3.0","member":"package/files/inter-latin-wght-normal.woff2","target":"inter.woff2","license_member":"package/LICENSE"},
    {"name":"@fontsource-variable/jetbrains-mono","version":"5.3.0","member":"package/files/jetbrains-mono-latin-wght-normal.woff2","target":"jetbrains-mono.woff2","license_member":"package/LICENSE"},
)


def main() -> None:
    """验证官方 NPM 归档完整性，仅复制明确列出的资源和许可证。

    :returns: 无；资源在包内 web/vendor，来源和许可证在 docs/vendor。
    """
    if not os.getenv("CONDA_PREFIX") or os.getenv("CONDA_DEFAULT_ENV")=="base":
        raise RuntimeError("请使用项目独立 Conda 环境")
    target=Path("src/yolo_agent/web/vendor")
    notices=Path("docs/vendor")
    target.mkdir(parents=True,exist_ok=True)
    notices.mkdir(parents=True,exist_ok=True)
    manifest=[]
    with httpx.Client(timeout=45,follow_redirects=True) as client:
        for package in PACKAGES:
            url=f"https://registry.npmjs.org/{package['name']}/{package['version']}"
            response=client.get(url)
            response.raise_for_status()
            metadata=response.json()
            archive_url=metadata["dist"]["tarball"]
            if urlparse(archive_url).hostname!="registry.npmjs.org" or urlparse(archive_url).scheme!="https":
                raise ValueError("前端资源只允许固定官方注册表")
            response=client.get(archive_url)
            response.raise_for_status()
            archive=response.content
            algorithm,encoded=metadata["dist"]["integrity"].split("-",1)
            if algorithm!="sha512" or hashlib.sha512(archive).digest()!=base64.b64decode(encoded):
                raise ValueError("官方资源归档完整性不符")
            with tarfile.open(fileobj=io.BytesIO(archive),mode="r:gz") as tar:
                # 不解压归档路径，只读取固定成员，避免引入脚本或写入任意文件。
                content=tar.extractfile(package["member"]).read()
                if "license_member" in package:
                    license_text=tar.extractfile(package["license_member"]).read()
                else:
                    response=client.get(package["license_url"])
                    response.raise_for_status()
                    license_text=response.content
            (target/package["target"]).write_bytes(content)
            license_path=notices/f"{package['name'].lstrip('@').replace('/','__')}.LICENSE.txt"
            license_path.write_bytes(license_text)
            manifest.append({"package":package["name"],"version":package["version"],"source":url,"archive":archive_url,"integrity":metadata["dist"]["integrity"],"file":package["target"],"sha256":hashlib.sha256(content).hexdigest(),"license":str(license_path),"license_sha256":hashlib.sha256(license_text).hexdigest()})
    (notices/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"prepared":[item["package"]+"@"+item["version"] for item in manifest]},ensure_ascii=False))


if __name__=="__main__":
    main()
