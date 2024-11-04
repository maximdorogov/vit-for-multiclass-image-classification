from transformers import ViTImageProcessor, ViTForImageClassification
import torch
from typing import List
import numpy as np

class GrassClassificationModel:
    """
    A wrapper for ViT models for classification.
    """

    def __init__(
        self,
        model_path:str,
        use_gpu: bool = True,
        use_fp16: bool = False,
    ):

        self._use_gpu = use_gpu
        self._use_fp16 = use_fp16
        self._model = ViTForImageClassification.from_pretrained(model_path)
        self._transforms = ViTImageProcessor.from_pretrained(model_path)
        
        self._device = torch.device('cuda' if self._use_gpu else 'cpu')
        self._model.eval()

        if self._use_fp16:
            self._model = self._model.half()
        if use_gpu:
            self._model.to(self._device)
        
    def run_inference(self, images: List[np.ndarray]) -> List[int]:
        """

        Parameters
        ----------

        Returns
        -------
        class_labels: List[int]
        """
        with torch.no_grad():
            inputs = self._transforms(images)
            inputs = torch.as_tensor(np.array(inputs['pixel_values']))
            output = self._model(pixel_values=inputs.cuda())
            return output.logits.argmax(-1)

if __name__ == "__main__":

    from PIL import Image

    model = GrassClassificationModel(
        model_path='./model_wheights/vit_tiny/checkpoint')

    input_img_a = Image.open('./data/images/0a1c68ef9.png')
    input_img_b = Image.open('./data/images/0a33283c7.png')
    input = [img.convert('RGB') for img in [input_img_a, input_img_b]]

    output = model.run_inference(input)

    print(output)