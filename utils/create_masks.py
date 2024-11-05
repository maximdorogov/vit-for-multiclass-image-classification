import os
import argparse
import torch
from PIL import Image
from ultralytics import FastSAM
import pandas as pd
import cv2
import numpy as np
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# NOTE:
# This is simple script and its not intended to be used as production code
# Some docstrings could be incomplete and some typehinting could be missing.

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='Script to build the image clasification dataset',
            add_help=True,
        )
    parser.add_argument('-d', '--input_images_folder',
                        help=('Kaggle dataset folder'),
                        type=str,
                        required=True)
    parser.add_argument('-c', '--input_csv_file',
                        help=('output path for the generated csv file with ',
                              'images path and labels'),
                        type=str,
                        required=True)
    parser.add_argument('-of', '--output_mask_folder',
                        help=('Dataset output folder'),
                        type=str,
                        required=True)

    args = parser.parse_args()
    device = 'cuda'
    det_model_id = "IDEA-Research/grounding-dino-base"
    seg_model_id = 'FastSAM-x.pt'
    texts_prompt = "grass, green leaves, green areas."

    os.makedirs(args.output_mask_folder, exist_ok=True)

    processor = AutoProcessor.from_pretrained(det_model_id)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(
        det_model_id).to(device)
    
    sam = FastSAM(seg_model_id)

    dataset_df = pd.read_csv(args.input_csv_file, header=None)

    for img_name in dataset_df[0]:
        image = Image.open(
            os.path.join(args.input_images_folder, img_name)).convert("RGB")

        inputs = processor(
            images=image, text=texts_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = detector(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.25,
            text_threshold=0.2,
            target_sizes=[image.size[::-1]]
        )
        try:
            sam_results = sam(
                image, bboxes=results[0]['boxes'])
        except:
            print(f'cannot comput masks for image {img_name}')

        masked = image.copy()
        if any(sam_results):

            sam_result = sam_results[0]
            masks = sam_result.masks.data.cpu().numpy().squeeze()

            img = np.array(image, dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            mask_shape = img.shape[:2]
            final_mask = np.zeros(mask_shape, dtype=np.uint8)

            for mask in masks:
                if np.max(mask) == 0:
                    continue
                mask = np.array(mask, dtype=np.uint8)
                mask = cv2.resize(mask, dsize=mask_shape[::-1])
                print(mask.shape, final_mask.shape)
                final_mask = cv2.bitwise_or(final_mask, mask)
            masked = cv2.bitwise_and(img, img, mask=final_mask)
        
        cv2.imwrite(os.path.join(args.output_mask_folder,img_name), masked)