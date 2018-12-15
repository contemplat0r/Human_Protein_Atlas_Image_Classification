import os
import pathlib

import pandas as pd
import numpy as np
import matplotlib.image as mpimg
from sklearn.preprocessing import MultiLabelBinarizer

import torch
from torchvision import datasets, transforms, models
from torch.utils import data

classes = np.arange(0,28)
multilabel_binarizer = MultiLabelBinarizer(classes)
multilabel_binarizer.fit(classes)

INPUT_DIR = '../input'

def fill_targets(df_row, targets_column_name='Target'):
    df_row[targets_column_name] = np.array(df_row[targets_column_name].split(' ')).astype(np.int8)
    for label_name in LABEL_NAMES:
        df_row[label_name] = 0
    for num in df_row[targets_column_name]:
        df_row[LABEL_NAMES[int(num)]] = 1
    return df_row

class HumanProteinAtlasDataset(data.Dataset):

    #def __init__(self, images_color_channels_fnames, transform=None):
    def __init__(self, images_description_df, transform=None, train=True):

        #self.images_color_channels_fnames = images_color_channels_fnames
        self.images_description_df = images_description_df.copy()
        self.transform = transform
        self.train_mode = True

    def __len__(self):
        return len(images_color_channels_fnames.keys())


    def __getitem__(self, index):
        return 


    def _load_imgs_color_channels(self):
	self.imgs_color_chanels = {}
	for img_grp_id, img_fnames in self.images_color_channels_fnames.items():
	    grouped_color_channels = {
		img_fname[:-4]: mpimg.imread(pathlib.Path(INPUT_DIR, img_fname).as_posix()) for img_fname in img_fnames
	    }
	    imgs_color_chanels[img_grp_id] = grouped_color_channels
	return imgs_color_chanels
