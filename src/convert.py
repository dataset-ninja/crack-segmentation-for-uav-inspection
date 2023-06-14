# Path to the original dataset
import os

import numpy as np
import supervisely as sly
from supervisely.io.fs import get_file_name


def create_ann(image_path, ann_dir, obj_class):
    labels = []
    file_name = get_file_name(image_path)
    ann_path = os.path.join(ann_dir, file_name + ".jpg")
    image_np = sly.image.read(ann_path)[:, :, 0]
    img_height, img_wight = image_np.shape[0], image_np.shape[1]
    if len(np.unique(image_np)) != 1:
        mask = image_np == 255
        curr_bitmap = sly.Bitmap(mask)
        curr_label = sly.Label(curr_bitmap, obj_class)
        labels.append(curr_label)

    return sly.Annotation(img_size=(img_height, img_wight), labels=labels)


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    dataset_path = "../datasets-bot/datasets/crack-segmentation-for-uav-inspection"
    ds_name = "ds0"
    imd_dir = "images"
    ann_dir = "masks"

    # Function should read local dataset and upload it to Supervisely project, then return project info.
    obj_class = sly.ObjClass("crack", sly.Bitmap)
    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(obj_classes=[obj_class])
    api.project.update_meta(project.id, meta.to_json())

    dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

    images_path = os.path.join(dataset_path, imd_dir)
    ann_path = os.path.join(dataset_path, ann_dir)
    images_names = os.listdir(images_path)

    progress = sly.Progress(f"Process dataset {ds_name}", len(images_names))

    for images_names_batch in sly.batched(images_names):
        img_pathes_batch = [
            os.path.join(images_path, image_name) for image_name in images_names_batch
        ]

        img_infos = api.image.upload_paths(dataset.id, images_names_batch, img_pathes_batch)
        img_ids = [im_info.id for im_info in img_infos]

        anns = [create_ann(image_path, ann_path, obj_class) for image_path in img_pathes_batch]
        api.annotation.upload_anns(img_ids, anns)

        progress.iters_done_report(len(images_names_batch))

    return project
