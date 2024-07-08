"""
General Utility functions
"""
import os
import torch
from omegaconf import DictConfig
from .images_utils import image_to_mask_name


def create_directory(path):
    """
    Create Directory if it already does not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def join_paths(*paths):
    """
    Concatenate multiple paths.
    """
    return os.path.normpath(os.path.sep.join(path.rstrip(r"\/") for path in paths))


def set_gpus(gpu_ids):
    """
    Change number of visible GPUs for PyTorch.
    gpu_ids: Could be integer or list of integers.
    In case Integer: if integer value is -1 then use all available GPUs.
    otherwise if positive number, then use given number of GPUs.
    In case list of Integer: each integer will be considered as GPU id.
    """
    all_gpus = list(range(torch.cuda.device_count()))
    all_gpus_length = len(all_gpus)

    if isinstance(gpu_ids, int):
        if gpu_ids == -1:
            gpu_ids = all_gpus
        else:
            gpu_ids = list(range(min(gpu_ids, all_gpus_length)))
    elif isinstance(gpu_ids, list):
        gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id < all_gpus_length]

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

def get_gpus_count():
    """
    Return length of available GPUs.
    """
    return torch.cuda.device_count()


def get_data_paths(cfg: DictConfig, mode: str, mask_available: bool):
    """
    Return list of absolute images/mask paths.
    There are two options you can either pass directory path or list.
    In case of directory, it should contain relative path of images/mask
    folder from project root path.
    In case of list of images, every element should contain absolute path
    for each image and mask.
    For prediction, you can set mask path to None if mask are not
    available for visualization.
    """

    # read images from directory
    if isinstance(cfg.DATASET[mode].IMAGES_PATH, str):
        # has only images name not full path
        images_paths = os.listdir(
            join_paths(
                cfg.WORK_DIR,
                cfg.DATASET[mode].IMAGES_PATH
            )
        )

        if mask_available:
            mask_paths = [
                image_to_mask_name(image_name) for image_name in images_paths
            ]
            # create full mask paths from folder
            mask_paths = [
                join_paths(
                    cfg.WORK_DIR,
                    cfg.DATASET[mode].MASK_PATH,
                    mask_name
                ) for mask_name in mask_paths
            ]

        # create full images paths from folder
        images_paths = [
            join_paths(
                cfg.WORK_DIR,
                cfg.DATASET[mode].IMAGES_PATH,
                image_name
            ) for image_name in images_paths
        ]
    else:
        # read images and mask from absolute paths given in list
        images_paths = list(cfg.DATASET[mode].IMAGES_PATH)
        if mask_available:
            mask_paths = list(cfg.DATASET[mode].MASK_PATH)

    if mask_available:
        return images_paths, mask_paths
    else:
        return images_paths,
