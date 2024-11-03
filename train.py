import pandas as pd
import os
import argparse
from pydantic import BaseModel
from PIL import Image
import torch
import numpy as np
from torchvision import transforms as T
from torchvision.transforms import Normalize
from transformers import (
    ViTImageProcessor, 
    ViTForImageClassification,
    TrainingArguments, 
    Trainer,
)
import evaluate
import supervision as sv


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
        print(self.label2id)

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


class TrainingConfig(BaseModel):
    model_name: str
    epochs: int
    learning_rate: float


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Script to build the image clasification dataset',
            add_help=True,
        )
    parser.add_argument('-d', '--input_images_path',
                        help=('A folder with images used for the dataset'),
                        type=str,
                        required=False)
    parser.add_argument('-if', '--input_csv_file',
                        help=('csv file containing image names in the first ',
                              'column and image classes in the second'),
                        type=str,
                        required=True)
    parser.add_argument('-e', '--experiment_folder',
                        help=('Folder to save the model and training artifacts'),
                        type=str,
                        required=False)
    parser.add_argument('-m', '--model_name',
                        help=('Pretrained ViT model name compatible with ',
                              'transformers python package'),
                        type=str,
                        default='WinKawaks/vit-small-patch16-224',
                        required=False)

    args = parser.parse_args()

    # set seeds
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    model_name = args.model_name
    # create experiment folder
    os.makedirs(args.experiment_folder, exist_ok=True)

    processor = ViTImageProcessor.from_pretrained(model_name)
    transforms = T.Compose([
        T.ToTensor(),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.Resize((processor.size["height"], processor.size["width"])),
        Normalize(mean=processor.image_mean, std=processor.image_std)
    ])

    main_dataset = PlantGrassDataset(
        dataset_csv=args.input_csv_file,
        folder_path=args.input_images_path,
        transform=transforms,
        )

    # split main dataset
    val_perc = 0.2
    train_data, val_data = torch.utils.data.random_split(
        main_dataset, [1 - val_perc, val_perc])

    # export image filenames and classes used for validation
    data_for_export = [
        (val_data.dataset.image_name, val_data.dataset.class_name) 
        for _, _ in val_data]

    output_path = os.path.join(args.experiment_folder, 'val_data.csv')
    data_for_export = pd.DataFrame(data_for_export)
    data_for_export.to_csv(output_path, index=False, header=False)

    print(f'Train samples: {len(train_data)}\nTest samples: {len(val_data)}')

    # Training stage

    model = ViTForImageClassification.from_pretrained(
        model_name,
        id2label=main_dataset.id2label,
        label2id=main_dataset.label2id,
        ignore_mismatched_sizes=True)
    
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir=args.experiment_folder,
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=4,
        num_train_epochs=10,
        warmup_ratio=0.1,
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=processor,
        compute_metrics=compute_metrics,
    )
    trainer.train()

