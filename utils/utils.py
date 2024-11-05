import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from typing import Any, Tuple
from typing_extensions import Annotated
from pydantic import BaseModel, Field
import evaluate
import yaml


class TrainingConfig(BaseModel):
    model_name: str = 'WinKawaks/vit-small-patch16-224'
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 5e-4
    data_split: Annotated[float, Field(gt=0, lt=1)]


def collate_fn(data: Any):
    """
    Collator function to properly pass raw data without preprocessing
    """
    return tuple(map(list, zip(*data)))

def load_training_cfg(cfg_path:str) -> TrainingConfig:
    """
    Loads training config file.

    Parameters
    ----------
    cfg_path:str
        Path to a trainer config yaml file
    
    Returns
    -------
    TrainingConfig object
    """
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return TrainingConfig(**cfg)

def compute_metrics(eval_pred: Tuple[Any, Any]):
    """
    Helper function for evaluation metric computation.
    """
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

class PlantGrassDataset(torch.utils.data.Dataset):

    IMG_CLASS_COL = 1

    def __init__(self, dataset_csv:str, folder_path: str, transform=None):
        """
        Parameters
        ----------
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
    ):
        """
        Parameters
        ----------
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