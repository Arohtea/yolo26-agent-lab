"""复用图像与科学计算库检查数据质量，不推断未知类别。"""

import hashlib
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
import networkx as nx
import numpy as np
from PIL import Image, ImageOps
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupShuffleSplit

FORMATS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP"}


def inspect_image(path: Path, preview: Path, max_pixels: int, *, allow_mpo_primary_frame: bool = False) -> dict:
    """校验文件结构并完整解码，生成面向人工审核的定向预览。

    :param path: 平台已经接管的原始图片。
    :param preview: 平台生成的 JPEG 预览路径。
    :param max_pixels: 单张图片允许的解码像素上限。
    :param allow_mpo_primary_frame: 已标注航拍 JPEG/MPO 仅校验和使用主画面，保留源文件。
    :returns: 校验状态、原始文件指纹、显示尺寸及感知哈希。
    """
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    result = {"sha256": digest, "status": "corrupt"}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path, formats=list(FORMATS.values())) as original:
                if original.width * original.height > max_pixels:
                    raise ValueError(f"图片超过 {max_pixels} 像素上限")
                mpo = allow_mpo_primary_frame and original.format == "MPO" and path.suffix.lower() in {".jpg", ".jpeg"}
                if original.format != FORMATS[path.suffix.lower()] and not mpo:
                    raise ValueError("真实图片编码与扩展名不一致")
                result["format"] = original.format
                original.verify()
            # verify 不解码像素，重新打开并 load 才能发现截断的压缩图像。
            with Image.open(path, formats=list(FORMATS.values())) as original:
                original.load()
                if getattr(original, "is_animated", False) and not (allow_mpo_primary_frame and original.format == "MPO"):
                    raise ValueError("第一版仅接收静态图片，请先导出独立帧")
                result["exif_orientation"] = int(original.getexif().get(274, 1))
                oriented = ImageOps.exif_transpose(original)
                result.update(width=oriented.width, height=oriented.height, phash=str(imagehash.phash(oriented, hash_size=8)))
                oriented.thumbnail((512, 512), Image.Resampling.LANCZOS)
                oriented.convert("RGB").save(preview, format="JPEG", quality=85)
        result["status"] = "valid"
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"[:2000]
    return result


def analyze_images(images: list[dict], threshold: int, max_pairs: int) -> tuple[dict, nx.Graph]:
    """生成可复核的质量报告与重复关系图。

    :param images: 已冻结范围的图片记录。
    :param threshold: 64 位 pHash 的最大汉明距离。
    :param max_pairs: 不同感知哈希之间允许的候选对上限。
    :returns: 完整质量报告和包含所有有效图片的重复关系图。
    :raises ValueError: 候选数量超出可安全处理的限制。
    """
    valid = [image for image in images if image["status"] == "valid"]
    graph = nx.Graph()
    graph.add_nodes_from(image["id"] for image in valid)
    hashes: dict[str, list[str]] = defaultdict(list)
    exact: dict[str, list[str]] = defaultdict(list)
    sha_by_id = {}
    for image in valid:
        hashes[image["phash"]].append(image["id"])
        exact[image["sha256"]].append(image["id"])
        sha_by_id[image["id"]] = image["sha256"]
    for members in [*exact.values(), *hashes.values()]:
        graph.add_edges_from((members[0], member) for member in members[1:])

    keys = sorted(hashes)
    pair_count = 0
    if len(keys) > 1:
        bits = np.array([imagehash.hex_to_hash(key).hash.flatten() for key in keys], dtype=np.uint8)
        tree = cKDTree(bits)
        pair_count = int((tree.count_neighbors(tree, r=threshold, p=1) - len(keys)) // 2)
        if pair_count > max_pairs:
            raise ValueError(f"近似候选对 {pair_count} 超过上限 {max_pairs}，请调低阈值或拆分数据集")
        for left, right in tree.query_pairs(r=threshold, p=1, output_type="ndarray"):
            graph.add_edge(hashes[keys[left]][0], hashes[keys[right]][0])

    near_groups = []
    for members in nx.connected_components(graph):
        if len({sha_by_id[member] for member in members}) > 1:
            near_groups.append(sorted(members))
    resolution = Counter(f"{image['width']}x{image['height']}" for image in valid)
    widths = [image["width"] for image in valid]
    heights = [image["height"] for image in valid]
    report = {
        "schema_version": 1, "total_images": len(images), "valid_images": len(valid),
        "status_counts": dict(Counter(image["status"] for image in images)),
        "corrupt_images": [{"image_id": image["id"], "name": image["original_name"], "error": image["error"]} for image in images if image["status"] != "valid"],
        "format_counts": dict(Counter(image["format"] for image in valid)),
        "resolution_counts": dict(sorted(resolution.items())),
        "dimensions": {"width_min": min(widths) if widths else None, "width_max": max(widths) if widths else None, "height_min": min(heights) if heights else None, "height_max": max(heights) if heights else None},
        "exact_duplicate_groups": [members for members in exact.values() if len(members) > 1],
        "exact_duplicate_excess": sum(len(members) - 1 for members in exact.values()),
        "near_duplicate_groups": sorted(near_groups),
        "similarity": {"algorithm": "imagehash.phash", "hash_bits": 64, "hamming_threshold": threshold, "distinct_hash_pairs": pair_count, "meaning": "候选关联分量，可能误报；不代表类别相同"},
        "annotation_status": "unlabeled", "class_distribution": None,
        "class_distribution_reason": "原始图片检查不推断类别；经人工审核后，由封存工件提供真实类别统计",
        "pixel_orientation": "exif_transposed", "ready_for_training": False,
    }
    return report, graph


def split_images(images: list[dict], graph: nx.Graph, policy: dict) -> tuple[list[dict], dict]:
    """先合并采集关系，再按关联分量划分，避免相邻帧与重复图泄漏。

    :param images: 版本创建时冻结的全部图片记录。
    :param graph: 数据检查阶段产生的重复关系图。
    :param policy: 已通过 VersionCreate 校验的划分参数。
    :returns: 成员快照与实际划分、排除摘要。
    :raises ValueError: 分组不足或比例无法生成三个非空子集。
    """
    valid = [image for image in images if image["status"] == "valid"]
    for field in policy["group_by"]:
        first: dict[str, str] = {}
        for image in valid:
            value = image[field]
            if value:
                if value in first:
                    graph.add_edge(first[value], image["id"])
                else:
                    first[value] = image["id"]
    groups = {}
    for members in nx.connected_components(graph):
        canonical = min(members)
        groups.update({member: canonical for member in members})

    # 完全相同的文件只保留一个；关联分组在去重之前构建，以保留所有采集约束。
    seen = set()
    selected = []
    excluded_duplicates = []
    for image in valid:
        if image["sha256"] in seen:
            excluded_duplicates.append(image["id"])
        else:
            selected.append(image)
            seen.add(image["sha256"])
    group_ids = np.array([groups[image["id"]] for image in selected])
    component_count = len(set(group_ids))
    if component_count < 3:
        raise ValueError(f"仅有 {component_count} 个独立关联分组，无法安全划分 train/val/test；请补充独立采集数据，不要拆散重复图或相邻帧")
    try:
        indices = np.arange(len(selected))
        train_val, test = next(GroupShuffleSplit(n_splits=1, test_size=policy["test_ratio"], random_state=policy["seed"]).split(indices, groups=group_ids))
        relative_val = policy["val_ratio"] / (policy["train_ratio"] + policy["val_ratio"])
        train_local, val_local = next(GroupShuffleSplit(n_splits=1, test_size=relative_val, random_state=policy["seed"]).split(train_val, groups=group_ids[train_val]))
    except ValueError as exc:
        raise ValueError("当前组数与比例无法生成三个非空子集，请增加独立组或调整比例") from exc
    snapshots = []
    counts = {}
    for split, positions in (("train", train_val[train_local]), ("val", train_val[val_local]), ("test", test)):
        counts[split] = len(positions)
        for position in positions:
            image = selected[int(position)]
            snapshots.append({**image, "split": split, "group_id": groups[image["id"]]})
    snapshots.sort(key=lambda image: image["id"])
    return snapshots, {
        "image_count": len(selected), "independent_groups": component_count,
        "split_counts": counts,
        "actual_image_ratios": {name: count / len(selected) for name, count in counts.items()},
        "excluded_exact_duplicate_ids": excluded_duplicates,
        "excluded_invalid_ids": [image["id"] for image in images if image["status"] != "valid"],
        "class_distribution": None, "annotation_status": "unlabeled", "ready_for_training": False,
        "warnings": ["比例作用于关联分组，实际图片比例可能偏离目标", "缺失采集元数据时无法仅凭图像保证相邻帧不泄漏", "近似重复是候选关系，当前保守分配到同一子集；未自动删除"],
    }
