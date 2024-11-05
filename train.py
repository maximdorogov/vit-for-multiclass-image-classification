import pandas as pd
import os
import argparse
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
from utils.utils import PlantGrassDataset, load_training_cfg, compute_metrics


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
    parser.add_argument('-t', '--train_cfg',
                        help=('Training config file'),
                        type=str,
                        required=True)

    args = parser.parse_args()

    # set seeds
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    train_cfg = load_training_cfg(cfg_path=args.train_cfg)

    # create experiment folder
    os.makedirs(args.experiment_folder, exist_ok=True)

    model_name = train_cfg.model_name
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
    val_perc = train_cfg.data_split
    train_data, val_data = torch.utils.data.random_split(
        main_dataset, [1 - val_perc, val_perc])

    # Export image filenames and classes used for validation
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

    training_args = TrainingArguments(
        output_dir=args.experiment_folder,
        remove_unused_columns=False,
        #  Eval is done at the end of each epoch.
        evaluation_strategy="epoch",
        # Save after every epoch if metric improves
        save_strategy="epoch",
        learning_rate=train_cfg.learning_rate,
        per_device_train_batch_size=train_cfg.batch_size,
        # steps before performing a backward/update pass.
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=train_cfg.batch_size,
        num_train_epochs=train_cfg.epochs,
        # Ratio of training steps used for a linear warmup from 0 to lr
        warmup_ratio=0.1,
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
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

