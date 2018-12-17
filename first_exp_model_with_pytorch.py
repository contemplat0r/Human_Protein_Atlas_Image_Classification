import os
import pathlib

import pandas as pd
import numpy as np
import matplotlib.image as mpimg
from sklearn.preprocessing import MultiLabelBinarizer
from PIL import Image

import torch
from torchvision import datasets, transforms, models
from torch.utils import data

CLASSES = np.arange(0,28)
multilabel_binarizer = MultiLabelBinarizer(CLASSES)
multilabel_binarizer.fit(CLASSES)

INPUT_DIR = '../input'
TRAIN_IMAGES_DIR = pathlib.Path(INPUT_DIR, 'train').as_posix()
TEST_IMAGES_DIR = pathlib.Path(INPUT_DIR, 'test').as_posix()
TARGETS_COLUMN_NAME = 'Target'
COLORS = ('red', 'green', 'blue', 'yellow')
IMAGE_FILE_EXT = 'png'

class HumanProteinAtlasDataset(data.Dataset):

    def __init__(self, images_description_df, transform=None, train_mode=True):

        self.images_description_df = images_description_df.copy()
        self.transform = transform
        self.train_mode = train_mode
        if train_mode:
            self.path_to_img_dir = TRAIN_IMAGES_DIR
        else:
            self.path_to_img_dir = TEST_IMAGES_DIR

    def __len__(self):
        return self.images_description_df.shape[0]


    def __getitem__(self, index):
        multilabel_target = None
        color_image = self._load_multicolor_image(index)
        if self.train_mode:
            multilabel_target = self._load_multilabel_target(index)
        if self.transform:
                color_image = self.transform(color_image)
        return color_image, multilabel_target

    def _load_multicolor_image(self, index):
        img_components_id = self.images_description_df.iloc[index]['Id']
        print("_load_multicolor_image, img_components_id: ", img_components_id)
        image_color_components = []
        for color in COLORS:
            path_to_color_component_file = pathlib.Path(
                    self.path_to_img_dir, '{}_{}.{}'.format(
                        img_components_id, color, IMAGE_FILE_EXT
                    )
                )
            image_color_components.append(Image.open(path_to_color_component_file))
        return Image.merge('RGBA', bands=image_color_components) 

    def _load_multilabel_target(self, index):
        return multilabel_binarizer.transform(
                [
                    np.array(
                        self.images_description_df[TARGETS_COLUMN_NAME].iloc[index].split(' ')
                    ).astype(np.int8)
                ]
            )
