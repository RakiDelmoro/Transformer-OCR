from torch.utils.data import Dataset
from datasets.data_utils import text_to_image, text_to_tensor_and_pad_with_zeros
import pandas
import os
import numpy as np
import torch
import cv2 as cv

from constants import MAX_PHRASE_LENGTH

class OCRDataset(Dataset):

    def __init__(self, dataset_file_name, csv_file_name, image_file_name,
                 ):
        self.data_frame = pandas.read_csv(os.path.join(dataset_file_name, csv_file_name))

        self.dataset_file = dataset_file_name
        self.image_file = image_file_name
        self.max_epoch = 1000

    def __len__(self):
        return min(len(self.data_frame), self.max_epoch)
        # return len(self.data_frame)

    def __getitem__(self, index):

        if index >= self.max_epoch:
            raise IndexError()
        image_name = self.data_frame.iloc[index, 0]
        target = self.data_frame.iloc[index, -1]

        # Get the image file path
        image_file_path = os.path.join(self.dataset_file, self.image_file, image_name)
        # Turn into array
        image_to_np_aray = cv.imread(image_file_path)
        # Get rid of channel
        image = cv.cvtColor(image_to_np_aray, cv.COLOR_BGR2GRAY)
        # normalize pixel value to 0-1
        normalized_image = cv.normalize(image, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)

        target_as_tensor = text_to_tensor_and_pad_with_zeros(text=target, max_length=MAX_PHRASE_LENGTH)

        sample = {"image": torch.from_numpy(normalized_image), "expected": target_as_tensor}

        return sample
