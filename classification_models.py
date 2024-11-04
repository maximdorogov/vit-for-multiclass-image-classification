from transformers import ViTImageProcessor, ViTForImageClassification
import torch
from typing import List, Union, Any, Tuple
from PIL.Image import Image
import numpy as np


class GrassClassificationModel:

    def __init__(
        self,
        model_path:str,
        use_gpu: bool = True,
        use_fp16: bool = False,
    ):
        """
        A wrapper for ViT models for image classification.

        Parameters
        ----------
        model_path:str
            Path where model checkpoint is located.
        use_gpu: bool
            Flag to use nvidia gpu. Enabled by default.
        use_fp16: bool
            Use quantized model to speed up inferece time. Can decrease
            model accuracy.
        """
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
        
    def run_inference(
            self,
            images: Union[np.ndarray, List[np.ndarray], Image, List[Image]]
    ) -> Tuple[List[Any], List[float]]:
        """
        Runs inference on the given images.

        Parameters
        ----------
        images: Union[np.ndarray, List[np.ndarray], PIL.Image]
            Accepts a list of images, in BGR color space, as np.ndarray, a 
            single image or a PIL image object list.
        Returns
        -------
        Tuple[List[Any], List[float]] containing class labels and scores.
        """
        with torch.no_grad():
            inputs = self._transforms(images)
            inputs = torch.as_tensor(np.array(inputs['pixel_values']))
            output = self._model(pixel_values=inputs.to(self._device))
            preds = torch.max(output.logits.softmax(-1), dim=-1)
            labels = [
                self._model.config.id2label[id] 
                for id in preds.indices.cpu().tolist()]
            return labels, preds.values.cpu().tolist()
