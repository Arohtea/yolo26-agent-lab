"""操作者显式准备白名单 YOLO26 预训练权重与绘图字体。"""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

MODEL_REPO = "Ultralytics/YOLO26"
MODEL_REVISION = "070ac3c51435984d992ba92eb3f234daf4e4500d"


def main() -> None:
    """只获取固定仓库、固定提交与白名单文件，不暴露任意下载接口。

    :returns: 无；权重、字体与来源指纹保存到项目存储。
    """
    if not os.getenv("CONDA_PREFIX") or os.getenv("CONDA_DEFAULT_ENV") == "base":
        raise RuntimeError("请进入项目独立 Conda 环境")
    parser = argparse.ArgumentParser(description="准备固定版本 YOLO26 模型")
    parser.add_argument("--model", choices=["yolo26n.pt","yolo26s.pt","yolo26m.pt"], default="yolo26n.pt")
    args = parser.parse_args()
    from yolo_agent.settings import get_settings
    from yolo_agent.services.ultralytics_runtime import configure_ultralytics

    settings = get_settings()
    path = Path(hf_hub_download(repo_id=MODEL_REPO, revision=MODEL_REVISION, filename=args.model, local_dir=settings.training_model_dir))
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream,"sha256").hexdigest()
    provenance = {"model":args.model,"repo_id":MODEL_REPO,"revision":MODEL_REVISION,"sha256":digest}
    path.with_suffix(".provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    configure_ultralytics(settings)
    from ultralytics.utils import USER_CONFIG_DIR
    from ultralytics.utils.checks import check_font

    for name in ("Arial.ttf","Arial.Unicode.ttf"):
        font = check_font(name)
        if font:
            target = USER_CONFIG_DIR / name
            if Path(font).resolve() != target.resolve():
                shutil.copyfile(font, target)
    print(json.dumps(provenance,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
