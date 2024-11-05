import os
import pandas as pd
import argparse
import shutil

# NOTE:
# This is simple script and its not intended to be used as production code
# Some docstrings could be incomplete and some typehinting could be missing.


def is_image(file_name: str) -> bool:
    return file_name.lower().endswith(
        ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))    

def generate_image_csv(
    data_dir: str,
) -> pd.DataFrame:
    """
    Generate a CSV file with image paths and class labels.
    
    Parameters
    ----------
    data_dir:str
        Path to the root directory containing class-named folders of images.
    output_csv:str
        Path to save the generated CSV file.
    """
    data = []

    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)

        if os.path.isdir(class_path):
            for image_name in os.listdir(class_path):
                if is_image(file_name=image_name):
                    data.append([image_name, class_name])
    return pd.DataFrame(data)

def move_images_to_single_folder(
    data_dir: str, 
    output_folder: str
):
    """
    Move all images from class-named subfolders into a single output folder.
    
    Parameters
    ----------
    data_dir:str
        Path to the root directory containing class-named folders of images.
    output_folder:str
        Path to the folder where all images will be moved.
    """

    os.makedirs(output_folder, exist_ok=True)
    
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)

        if os.path.isdir(class_path):
            for image_name in os.listdir(class_path):

                if is_image(file_name=image_name):
                    src_path = os.path.join(class_path, image_name)
                    dest_path = os.path.join(output_folder, image_name)
                    shutil.move(src_path, dest_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='Script to build the image clasification dataset',
            add_help=True,
        )
    parser.add_argument('-d', '--input_data_path',
                        help=('Kaggle dataset folder'),
                        type=str,
                        required=True)
    parser.add_argument('-c', '--output_csv_file',
                        help=('output path for the generated csv file with ',
                              'images path and labels'),
                        type=str,
                        required=True)
    parser.add_argument('-of', '--output_image_folder',
                        help=('Dataset output folder'),
                        type=str,
                        required=True)

    args = parser.parse_args()

    images_df = generate_image_csv(data_dir=args.input_data_path)

    move_images_to_single_folder(
        data_dir=args.input_data_path, output_folder=args.output_image_folder)
    
    print(f'{len(images_df)} images moved')

    images_df.to_csv(args.output_csv_file, index=False, header=False)
