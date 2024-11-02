import pandas as pd
import os
import argparse
from typing import Tuple
from PIL import Image
import torch
import supervision as sv
from transformers import ViTImageProcessor, ViTForImageClassification

def get_labels(dataset_csv: str):
    dataset_pd = pd.read_csv(dataset_csv, header=None)
    return set(dataset_pd[1])

class PlantGrassDataset(torch.utils.data.Dataset):
    
    IMG_NAME_COL = 0
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
        image_name, class_name = self._dataset_df.loc[index]
        image = Image.open(os.path.join(self._folder_path, image_name))
        return image, self.label2id[class_name]

    def __len__(self):
        return len(self._dataset_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Script to build the image clasification dataset',
            add_help=True,
        )
    parser.add_argument('-d', '--input_images_path',
                        help=('Kaggle dataset folder'),
                        type=str,
                        required=False)
    parser.add_argument('-if', '--input_csv_file',
                        help=('output path for the generated csv file with',
                              ' images path and labels'),
                        type=str,
                        required=True)
    parser.add_argument('-of', '--training_cfg_filexs',
                        help=('Dataset output folder'),
                        type=str,
                        required=False)

    args = parser.parse_args()

    main_dataset = PlantGrassDataset(
        dataset_csv=args.input_csv_file,
        folder_path=args.input_images_path
        )

    image, label = main_dataset[0]

    print(main_dataset.id2label[label])
    sv.plot_image(image)
    
    # Map each class to a taget id and vice versa
    # idx2label = {idx: label for idx, label in enumerate(labels)}
    # label2idx = {label: idx for idx, label in enumerate(labels)}
    # model = ViTForImageClassification.from_pretrained(
    #     'google/vit-base-patch16-224',
    #     id2label=idx2label,
    #     label2id=label2idx,
    #     ignore_mismatched_sizes=True)   