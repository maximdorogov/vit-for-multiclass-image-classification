# Vision Transformer for multiclass Plant Seedlings Classification

## Summary
This work consists in a dataset preparation, training and evaluation of an image classification system based on the vision transformer (ViT) architecture from [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) paper. The dataset was extracted from the [Plant Seedlings Classification kaggle](https://www.kaggle.com/c/plant-seedlings-classification) challenge. 

During the first part of the task a classical multilabel classification was applied with no aditional preprossesing to the input data. In the second part a new dataset was constructed using Zero Shot Segmentation order to get rid of backround features and focus the feature extraction only on the plants present in each image.

## Requirements


## Data preparation
The dataset provided by Kaggle has the following structure:
```
plant-seedlings-classification
├── test
└── train
    ├── Black-grass
    ├── Charlock
    ├── Cleavers
    ├── Common Chickweed
    ├── Common wheat
    ├── Fat Hen
    ├── Loose Silky-bent
    ├── Maize
    ├── Scentless Mayweed
    ├── Shepherds Purse
    ├── Small-flowered Cranesbill
    └── Sugar beet
```
> NOTE: The data from the original `test` folder was excluded from the final dataset since it only contains unlabeled images.

The dataset contains 4750 images distributed accross 12 categories:

![title](report/data_distribution.png)

For simplicity the dataset was restructured moving all images in a single folder and keeping a `.csv` file with image names and classes:

```
6a4ef17c2.png,Cleavers
0515bc601.png,Cleavers
0ac327873.png,Cleavers
...
```

In order to achieve this we have `create_dataset.py` script:

```python
python ../utils/create_dataset.py -d KAGGLE_DATASET -fo PATH_TO_OUTPUT_CSV -of PATH_TO_OUTPUT_IMG_FOLDER
```
This will generate the csv file mentioned above and will move all the images from the subfolders to a single directory.
> You can also download this dataset from [here]() and drop it into the root of this repository to reproduce all the experiments.

## Stage 1: Model training

According to [An Image is Worth 16x16...](https://arxiv.org/abs/2010.11929) Vision Transformer attains excellent results compared to state-of-the-art convolutional networks while requiring fewer resources to train. Thats why smallest ViT model from the family was selected for this task in order to validate the idea and keep the GPU hours low.

Used models:
    
* vit-tiny-patch16-224

To train a model run:

```sh
python train.py -if DATASET_CSV_FILE -d DATASET_IMAGE_FOLDER -e EXPERIMENT_OUTPUT_FOLDER
```

Were:

- `-if:` csv file listing all images and classes generated in the previous steps
- `-d:` Folder containing all the images from the dataset
- `-e:` Output Folder to store the experiment assets, trained model and metada.
- `-t`: Training config yaml file.


### Training Configuration

In order to keep the reproducibility in each experiment a `yaml` file was used to track some of the most important hyperparameters. Most of them are alredy logged by `transformers` trainer class into the experiment folder. The lr scheduller (linear decay), optimizer (AdamW) and loss function (Cross Entropy) were used by default.

The default parameters related to data batching and model type are:

```python
class TrainingConfig(BaseModel):
    model_name: str = 'WinKawaks/vit-tiny-patch16-224'
    epochs: int = 30
    batch_size: int = 4
    learning_rate: float = 5e-4
    data_split: Annotated[float, Field(gt=0, lt=1)]
```

EXPLAIN TRAINING SPLIT AND DATA AGUMENTATION

### Results

The final results for `vit-tiny-patch16-224`:
```
accuracy: 0.95
average precission: 0.95
average f1: 0.95
average recall: 0.94
```
Bellow you can find the evaluation metrics per each class:

Confusion matrix

![title](report/cfm_tiny.png) 

Classification report

```
                           precision    recall  f1-score   support

              Black-grass       0.74      0.61      0.67        56
                 Charlock       0.99      1.00      0.99        81
                 Cleavers       0.98      0.98      0.98        53
         Common Chickweed       0.97      0.97      0.97       119
             Common wheat       0.95      0.95      0.95        37
                  Fat Hen       0.99      0.99      0.99        98
         Loose Silky-bent       0.84      0.93      0.88       126
                    Maize       1.00      0.97      0.99        36
        Scentless Mayweed       0.97      0.97      0.97       118
          Shepherds Purse       1.00      1.00      1.00        50
Small-flowered Cranesbill       1.00      0.99      1.00       103
               Sugar beet       0.99      0.96      0.97        73
```



## Stage 2: Zero Shot Segmentation Preprocessing

In this stage the idea was to preprocess the input images to get rid of backround features and focus the feature extraction only on the plants present in each image (foreground).


Two approaches were considered, first the recently released SAM2 segmentator, but it turns to be too slow (even using a GPU) and it tended to oversegment the images so a lot of different heuristics for each grass type had to be defined to clean up the SAM2 output masks. 

The second aproach, and the one I selected, was using a smaller model from the same family: `FastSAM` in combination with a Zero Shot detector: `GroundingDINO`.
GroundingDINO is a zero shot detector and it was used to select ROIs (regions of iterest) in each image with grass seeblings on it. Those ROIs, marked with bboxes were passed as a prompt to FastSAM to generate the desired segmentation mask isolating grass pixels from the background.

Some examples of the preprocessed dataset:

![title](report/segm_grid.png)

As you can see some of the images are segmented flawosly but others present some artifacts that will impact in the classification accuracy. This is becouse of image resolution, in images of better quality this segmentation aproach worked very well but if the resolution is too low or the grass is too thin it didnt segment very well.

### Results

The ViT model was trained with exactly the same parameters as in the previous stage but using this new dataset.

