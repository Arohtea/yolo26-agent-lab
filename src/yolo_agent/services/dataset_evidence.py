"""工作台与 Agent 共用封存 train/val 文件读取，不扫描原图或测试集兜底。"""

import hashlib
from pathlib import Path

from yolo_agent.services.datasets import DatasetError, owned_path


def verified_export_file(settings, export, manifest: dict, relative: str) -> Path:
    root = owned_path(settings, export.directory)
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file() or relative not in manifest["sha256"]:
        raise DatasetError(409, "封存文件缺失、未登记或路径越界")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != manifest["sha256"][relative]:
            raise DatasetError(409, "封存文件指纹变化")
    return path


def split_image_paths(settings, export, manifest: dict, split: str) -> list[Path]:
    if split not in {"train", "val"}:
        raise PermissionError("Agent 仅可读取封存 train/val 图片，不能访问测试集")
    root = owned_path(settings, export.directory)
    index = verified_export_file(settings, export, manifest, f"{split}.txt")
    result, identities = [], set()
    for line in index.read_text().splitlines():
        if not line.strip():
            continue
        path = (root / line.strip()).resolve()
        if not path.is_relative_to(root / "images" / split):
            raise DatasetError(409, "封存图片清单路径越界")
        if path.stem in identities:
            raise DatasetError(409, "封存划分存在同编号图片，无法确定对应文件")
        identities.add(path.stem)
        result.append(path)
    return result


def verified_split_image(settings, export, manifest: dict, split: str, image_id: str) -> Path:
    root = owned_path(settings, export.directory)
    matches = [path for path in split_image_paths(settings, export, manifest, split) if path.stem == image_id]
    if not matches:
        raise DatasetError(404, "图片不在指定的封存划分中")
    return verified_export_file(settings, export, manifest, matches[0].relative_to(root).as_posix())
