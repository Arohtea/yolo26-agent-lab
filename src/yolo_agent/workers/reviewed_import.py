"""复用 Datumaro 导入外部已审核 YOLO 数据；不伪造 CVAT 审核记录。"""

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED

import datumaro as dm
import numpy as np
import yaml
from rq import get_current_job
from sqlalchemy import update
from sqlalchemy.orm import Session

from yolo_agent.models.annotation import Annotation, AnnotationRun, ClassSchema, DatasetExport, ImageReview, ReviewEvent
from yolo_agent.models.database import create_database_engine
from yolo_agent.models.dataset import Dataset, DatasetImage, DatasetTask, DatasetVersion, VersionImage, new_id, utc_now
from yolo_agent.schemas.dataset import ImageRead
from yolo_agent.services.annotations import annotation_digest, assert_reviewed
from yolo_agent.services.datasets import fail_dataset_task, owned_path
from yolo_agent.services.inspector import inspect_image, analyze_images
from yolo_agent.settings import get_settings


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,sort_keys=True,indent=2)+"\n")
    return digest(path)


def run_reviewed_import(task_id: str):
    """固定任务入口；已审认定来自操作者显式数据包确认，不执行新的人审。"""
    job = get_current_job()
    if job is None or job.id != task_id:
        raise RuntimeError("导入必须通过对应 RQ 任务执行")
    settings = get_settings()
    engine = create_database_engine(settings)
    try:
        with Session(engine) as session, session.begin():
            claimed = session.execute(update(DatasetTask).where(DatasetTask.id==task_id, DatasetTask.status=="queued").values(status="running",started_at=utc_now()))
            if claimed.rowcount != 1:
                raise RuntimeError("任务已执行或不存在，拒绝重复导入")
            task = session.get(DatasetTask, task_id)
            dataset_id, parameters = task.dataset_id, task.parameters
            project_id = session.get(Dataset, dataset_id).project_id
        if not parameters.get("accepted_reviewed_source") or not parameters.get("reviewer") or not parameters.get("confirmation"):
            raise ValueError("缺少操作者对现有已审来源的明确确认")
        source = owned_path(settings, parameters["source"])
        provenance = json.loads((source/"source_manifest.json").read_text())
        if not provenance.get("operator_authorized_import"):
            raise ValueError("来源包未声明已审核授权")
        for item in provenance["files"]:
            for key, hash_key in (("image","image_sha256"),("label","label_sha256")):
                path = (source/item[key]).resolve()
                if not path.is_relative_to(source.resolve()) or path.is_symlink() or digest(path)!=item[hash_key]:
                    raise ValueError("来源文件越界或指纹变化")
        # Datumaro 使用 images/<split> 与 labels/<split>；来源包可保留原来的 <split>/images 布局。
        view=owned_path(settings,f"imports/{task_id}-view")
        for item in provenance["files"]:
            original_split=Path(item["image"]).parts[0]
            split={"valid":"val"}.get(original_split,original_split)
            if split not in {"train","val","test"}:
                raise ValueError("来源包含未知划分")
            for key,folder in (("image","images"),("label","labels")):
                destination=view/folder/split/Path(item[key]).name
                destination.parent.mkdir(parents=True,exist_ok=True)
                os.link(source/item[key],destination)
        (view/"data.yaml").write_text(yaml.safe_dump({"train":"images/train","val":"images/val","test":"images/test","names":{int(i):n for i,n in provenance["classes"].items()}},sort_keys=False))
        imported = dm.Dataset.import_from(str(view), format="yolo")
        categories = imported.categories()[dm.AnnotationType.label].items
        classes = [{"class_id":i,"name":label.name,"prompt":label.name} for i,label in enumerate(categories)]
        if [c["name"] for c in classes] != [provenance["classes"][str(i)] for i in range(len(classes))]:
            raise ValueError("解析类别与来源清单不一致")
        items = list(imported)
        if len(items)!=provenance["image_count"]:
            raise ValueError("解析图片数量不等于来源清单")
        with Session(engine) as session, session.begin():
            session.get(DatasetTask,task_id).total=len(items)
        schema_id, version_id, run_id, export_id = new_id(),new_id(),new_id(),new_id()
        accepted_at = utc_now()
        attribution = {"source":"external_reviewed_yolo","reviewer":parameters["reviewer"],"confirmation":parameters["confirmation"],"accepted_at":accepted_at,"source_manifest_sha256":digest(source/"source_manifest.json"),"new_manual_review_performed":False}
        members, boxes, reviews, events, exported, clipping = [],[],[],[],[],[]
        split_counts, class_counts = Counter(),Counter()
        for index,item in enumerate(items,1):
            split = {"valid":"val","validation":"val"}.get(item.subset,item.subset)
            if split not in {"train","val","test"}:
                raise ValueError(f"未知来源划分：{item.subset}")
            path = Path(item.media.path).resolve()
            if not path.is_relative_to(view.resolve()):
                raise ValueError("图片媒体路径越界")
            relative = str(path.relative_to(settings.storage_dir.resolve()))
            image_id = new_id()
            preview = f"previews/{image_id}.jpg"
            information = inspect_image(path,owned_path(settings,preview),settings.max_image_pixels)
            if information["status"]!="valid":
                raise ValueError(f"图片未通过校验：{path.name} {information.get('error')}")
            image = DatasetImage(id=image_id,dataset_id=dataset_id,import_task_id=task_id,
                original_name=path.name,storage_path=relative,size_bytes=path.stat().st_size,
                preview_path=preview,**information)
            with Session(engine,expire_on_commit=False) as session, session.begin():
                session.add(image)
                session.get(DatasetTask,task_id).processed=index
            snapshot={**ImageRead.model_validate(image).model_dump(),"storage_path":relative,"split":split}
            members.append(snapshot)
            image_boxes, source_boxes, normalized_annotations = [],[],[]
            for number,annotation in enumerate(item.annotations):
                if annotation.type!=dm.AnnotationType.bbox:
                    raise ValueError("仅接收矩形检测标注")
                original=np.asarray(annotation.points,dtype=np.float64)
                bounded=np.clip(original,[0,0,0,0],[image.width,image.height,image.width,image.height])
                if not np.all(np.isfinite(bounded)) or bounded[2]<=bounded[0] or bounded[3]<=bounded[1]:
                    raise ValueError("标注框裁剪后为空或含非法数值")
                changed=not np.array_equal(original,bounded)
                if changed:
                    clipping.append({"image":str(path.relative_to(view)),"index":number,"before":original.tolist(),"after":bounded.tolist()})
                box=Annotation(id=new_id(),run_id=run_id,image_id=image_id,class_id=annotation.label,
                    class_name=classes[annotation.label]["name"],bbox=bounded.tolist(),confidence=None,
                    annotation_source="external_reviewed_yolo",annotation_model_version="operator-authorized-import",
                    review_status="approved",flags=["boundary_clipped_on_import"] if changed else [],reviewer=parameters["reviewer"])
                image_boxes.append(box);class_counts[box.class_name]+=1
                source_boxes.append({"id":box.id,"class_id":annotation.label,"class_name":box.class_name,"annotation_source":"external_reviewed_yolo","bbox":original.tolist()})
                normalized_annotations.append(dm.Bbox(bounded[0],bounded[1],bounded[2]-bounded[0],bounded[3]-bounded[1],label=annotation.label))
            boxes.extend(image_boxes)
            reviews.append(ImageReview(run_id=run_id,image_id=image_id,status="approved",missed_objects_checked=True,
                reviewer=parameters["reviewer"],annotation_digest=annotation_digest(image_boxes),reviewed_at=accepted_at))
            events.append(ReviewEvent(id=new_id(),run_id=run_id,image_id=image_id,reviewer=parameters["reviewer"],
                source="operator_authorized_reviewed_import",before=source_boxes,
                after=[{"class_id":b.class_id,"bbox":b.bbox,"authorization":attribution} for b in image_boxes]))
            exported.append(dm.DatasetItem(id=image_id,subset=split,media=dm.Image.from_file(str(path)),annotations=normalized_annotations))
            split_counts[split]+=1
        if len(boxes)!=sum(len(item.annotations) for item in items):
            raise ValueError("导入框数量变化")
        quality,_=analyze_images(members,4,settings.max_similarity_pairs)
        quality.update(annotation_status="external_reviewed",class_distribution=dict(class_counts),
            class_distribution_reason="来自操作者确认已审核的原始 YOLO 标签；本次未进行新的人工审核",ready_for_training=True)
        summary={"image_count":len(members),"annotation_count":len(boxes),"split_counts":dict(split_counts),"class_distribution":dict(class_counts),"reviewed_images":len(members),"unreviewed_images":0,"ready_for_training":True,"review_attribution":attribution,"source_split_preserved":True,"clipped_boxes":len(clipping),"warnings":["继承原数据划分；存在相邻帧或近似图跨集的风险，本批用于自主循环测试，不作泛化精度结论"]}
        manifest_relative=f"versions/{version_id}/manifest.json"
        version_digest=write_json(owned_path(settings,manifest_relative),{"version_id":version_id,"dataset_id":dataset_id,"number":1,"images":members,"summary":summary,"quality_report":quality,"review_attribution":attribution})
        version=DatasetVersion(id=version_id,dataset_id=dataset_id,task_id=task_id,number=1,name="外部已审核数据原划分",policy={"entry":"external_reviewed_yolo","preserve_source_splits":True},summary=summary,manifest_path=manifest_relative,manifest_sha256=version_digest,sealed=False)
        with Session(engine) as session, session.begin():
            session.add(ClassSchema(id=schema_id,project_id=project_id,name=f"已审 YOLO 原始{len(classes)}类",classes=classes))
            session.add(version);session.flush()
            session.add_all([VersionImage(version_id=version_id,image_id=m["id"],split=m["split"],group_id="external:"+m["id"],snapshot=m) for m in members]);session.flush()
            version.sealed=True
            session.add(AnnotationRun(id=run_id,version_id=version_id,class_schema_id=schema_id,status="reviewed",parameters={"review_attribution":attribution},model_version="external-reviewed-yolo",processed=len(members),total=len(members)));session.flush()
            session.add_all(boxes+reviews+events)
        export_relative=f"exports/{export_id}"
        target=owned_path(settings,export_relative)
        dm.Dataset.from_iterable(exported,categories=[c["name"] for c in classes]).export(str(target),"yolo_ultralytics",save_media=False)
        # 图片已通过逐文件指纹校验，硬链接只复用不可变文件，不重新压缩或复制像素。
        for item in exported:
            destination=target/"images"/item.subset/(item.id+Path(item.media.path).suffix)
            destination.parent.mkdir(parents=True,exist_ok=True)
            if not destination.exists():os.link(item.media.path,destination)
        write_json(target/"review-source.json",{"authorization":attribution,"clipped_boxes":clipping})
        hashes={str(p.relative_to(target)):digest(p) for p in sorted(target.rglob('*')) if p.is_file()}
        export_manifest={"export_id":export_id,"version_id":version_id,"run_id":run_id,"classes":classes,"summary":summary,"sha256":hashes,"reviewed_at":accepted_at}
        manifest_hash=write_json(target/"manifest.json",export_manifest)
        archive=target.with_suffix('.zip')
        with ZipFile(archive,'w',compression=ZIP_STORED,allowZip64=True) as zipped:
            for p in sorted(target.rglob('*')):
                if p.is_file():zipped.write(p,str(p.relative_to(target)))
        record=DatasetExport(id=export_id,version_id=version_id,run_id=run_id,directory=export_relative,manifest_sha256=manifest_hash,archive_path=str(archive.relative_to(settings.storage_dir.resolve())),archive_sha256=digest(archive),summary=summary)
        result={"dataset_id":dataset_id,"version_id":version_id,"annotation_run_id":run_id,"export_id":export_id,"summary":summary}
        report=f"reports/reviewed-import-{task_id}.json"
        write_json(owned_path(settings,report),result)
        with Session(engine) as session, session.begin():
            assert_reviewed(session,session.get(AnnotationRun,run_id))
            session.add(record);session.get(AnnotationRun,run_id).status="exported"
            task=session.get(DatasetTask,task_id);task.status="completed";task.finished_at=utc_now();task.result=result;task.report_path=report
        return result
    except Exception as error:
        fail_dataset_task(engine,task_id,f"{type(error).__name__}: {error}"[:4000])
        raise
    finally:
        engine.dispose()
