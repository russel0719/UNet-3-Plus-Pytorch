import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import numpy as np
from omegaconf import DictConfig

from utils.general_utils import get_data_paths
from utils.images_utils import prepare_image, prepare_mask

class DataGenerator(Dataset):
    """
    Generate batches of data for model by reading images and their
    corresponding masks.
    There are two options: you can either pass a directory path or a list.
    In case of directory, it should contain the relative path of images/mask
    folder from project root path.
    In case of list of images, every element should contain an absolute path
    for each image and mask.
    Because this generator is also used for prediction, so during testing you can
    set mask path to None if masks are not available for visualization.
    """

    def __init__(self, cfg: DictConfig, mode: str):
        """
        Initialization
        """
        self.cfg = cfg
        self.mode = mode
        self.batch_size = self.cfg.HYPER_PARAMETERS.BATCH_SIZE
        # set seed for reproducibility
        np.random.seed(cfg.SEED)

        # check if masks are available
        self.mask_available = False if cfg.DATASET[mode].MASK_PATH is None or str(
            cfg.DATASET[mode].MASK_PATH).lower() == "none" else True

        data_paths = get_data_paths(cfg, mode, self.mask_available)

        self.images_paths = data_paths[0]
        if self.mask_available:
            self.mask_paths = data_paths[1]

        self.on_epoch_end()
        self.__data_generation(self.indexes)

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.floor(len(self.images_paths) / self.batch_size))

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.images_paths))
        if self.cfg.PREPROCESS_DATA.SHUFFLE[self.mode].VALUE:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        """
        Generate one batch of data
        """

        if self.mask_available:
            return self.batch_images[index], self.batch_masks[index]
        else:
            return self.batch_images[index],

    def __data_generation(self, indexes):
        """
        Generates batch data
        """
        # create empty array to store batch data
        self.batch_images = []
        
        if self.mask_available:
            self.batch_masks = []

        for i, index in enumerate(indexes):
            # extract path from list
            img_path = self.images_paths[int(index)]
            if self.mask_available:
                mask_path = self.mask_paths[int(index)]

            # prepare image for model by resizing and preprocessing it
            image = prepare_image(
                img_path,
                self.cfg.PREPROCESS_DATA.RESIZE,
                self.cfg.PREPROCESS_DATA.IMAGE_PREPROCESSING_TYPE,
            )

            if self.mask_available:
                # prepare image for model by resizing and preprocessing it
                mask = prepare_mask(
                    mask_path,
                    self.cfg.PREPROCESS_DATA.RESIZE,
                    self.cfg.PREPROCESS_DATA.NORMALIZE_MASK,
                )

            # convert to PyTorch tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            if self.mask_available:
                mask = torch.from_numpy(mask).long()

            # add to batch
            self.batch_images.append(image)

            if self.mask_available:
                # convert mask into one hot vectors
                # height x width --> height x width x output classes
                mask = F.one_hot(mask, num_classes=self.cfg.OUTPUT.CLASSES).permute(2, 0, 1).float()
                self.batch_masks.append(mask)


def get_data_loader(cfg: DictConfig, mode: str):
    """
    Return data loader for the given configuration and mode
    """
    dataset = DataGenerator(cfg=cfg, mode=mode)
    return DataLoader(dataset, batch_size=cfg.HYPER_PARAMETERS.BATCH_SIZE, shuffle=cfg.PREPROCESS_DATA.SHUFFLE[mode].VALUE)
