import os
import random

import pathlib

import numpy as np
import pandas as pd

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


def show_data_dir_content(data_dir):
    return os.listdir(data_dir)

def load_text_data(pointer_to_text_data):
    return pd.read_csv(pointer_to_text_data)

def load_pickled_data(pointer_to_pickled_data):
    return 

def load_image_data(pointer_to_image_data):
    return

def select_objects(indexes_list, objects_names):
    return tuple(objects_names[i] for i in indexes_list)


def select_random_indexses_subset(size, subset_size):
    return random.sample(tuple(range(size)), subset_size)


def random_objects_select(objects_names, subset_size):
    objects_names_len = len(objects_names)
    indexes = select_random_indexses_subset(objects_names_len, subset_size)
    return select_objects(indexes, objects_names)


def select_offset_indexses_subset(size, subset_size, offset):
    return tuple(range(size))[offset:offset + subset_size]



def offset_objects_select(objects_names, subset_size, offset):
    objects_names_len = len(objects_names)
    indexes = select_offset_indexses_subset(objects_names_len, subset_size, offset)
    return select_objects(indexes, objects_names)

def read_and_group_dataset_fnames(
            path_to_dataset_dir,
            fname_pattern_begin,
            fname_pattern_end
        ):
    dataset_fnames = os.listdir(path_to_dataset_dir)
    grouped_fnames = {}
    for fname in dataset_fnames:
        path_to_file = os.path.join(path_to_datasets_dir, fname)
        fname_pattern = fname[fname_pattern_begin:fname_pattern_end]
        if fname_pattern not in grouped_fnames:
            grouped_fnames[fname_pattern] = [path_to_file]
        else:
            grouped_fnames[fname_pattern].append(path_to_file)
    return grouped_fnames 

#IMAGE_DIMENSIONS_NUM = 3
#images_dir = '../input/train'
#segmentation_file_path = '../input/train_ship_segmentations.csv'
#full_cwd_path = os.getcwd()
#path_prefix, cwd_itself = os.path.split(full_cwd_path)
#if cwd_itself != 'code':
#    os.chdir(os.path.join(path_prefix, 'code'))
#    print(os.getcwd())

#train_images_names = os.listdir(images_dir)

def group_img_fnames(img_group_ids, img_suffixs, img_ext):
    return {
        img_grp_id: ['{}_{}.{}'.format(
                img_grp_id,
                img_suffix,
                img_ext
            )
        for img_suffix in img_suffixs] for img_grp_id in img_group_ids
    }

def load_imgs_color_groups(grouped_img_fnames):
    imgs_color_groups = {}
    for img_grp_id, img_fnames in grouped_img_fnames.items():
        grouped_images = {
            img_fname[:-4]: mpimg.imread(pathlib.Path(INPUT_DIR, img_fname).as_posix()) for img_fname in img_fnames
        }
        imgs_color_groups[img_grp_id] = grouped_images
    return imgs_color_groups

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
        #print("_load_multicolor_image, img_components_id: ", img_components_id)
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
