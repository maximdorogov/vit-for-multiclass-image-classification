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

According to [An Image is Worth 16x16...](https://arxiv.org/abs/2010.11929) Vision Transformer attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train. Thats why  some of the smallest ViT's were selected for this task in order to keep the GPU hours low.

Used models:
    
* vit-tiny-patch16-224
* vit-small-patch16-224

### Training Configuration



## Stage 2: Zero Shot Segmentation Preprocessing

