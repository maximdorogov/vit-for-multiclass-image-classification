import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from typing import Tuple


class PlantGrassDataset(torch.utils.data.Dataset):

    IMG_CLASS_COL = 1

    def __init__(self, dataset_csv:str, folder_path: str, transform=None):
        """
        dataset_csv:str
            Path to a csv file contaning image name and class labels
        folder_path:str
            Path to a folder with all the images
        """
        super().__init__()

        self.transform = transform
        self._folder_path = folder_path
        self._dataset_df = pd.read_csv(dataset_csv, header=None)
        self.labels = list(set(self._dataset_df[self.IMG_CLASS_COL]))
        self.label2id = {label:i for i, label in enumerate(self.labels)}
        self.id2label = {i:label for i, label in enumerate(self.labels)}

    def __getitem__(self, index):
        # store as attributes for further usage
        self.image_name, self.class_name = self._dataset_df.loc[index]
        image = Image.open(os.path.join(self._folder_path, self.image_name))
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return {"pixel_values": image, "label": self.label2id[self.class_name]}

    def __len__(self):
        return len(self._dataset_df)
    

class EvalDataset(torch.utils.data.Dataset):
    """
    Dataset class to efficiently load data for model evaluation.
    """

    def __init__(
        self, 
        dataset_csv: str, 
        folder_path: str, 
        patch_shape: Tuple[int, int] = (224, 224)):
        """
        dataset_csv:str
            Path to a csv file contaning image name and class labels
        folder_path:str
            Path to a folder with all the images
        """
        super().__init__()

        self._folder_path = folder_path
        self._dataset_df = pd.read_csv(dataset_csv, header=None)

    def __getitem__(self, index):

        image_name, class_name = self._dataset_df.loc[index]
        image = Image.open(os.path.join(self._folder_path, image_name))

        return np.asarray(image.convert('RGB')), class_name

    def __len__(self):
        return len(self._dataset_df)