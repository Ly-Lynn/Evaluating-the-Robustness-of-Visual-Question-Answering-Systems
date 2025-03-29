from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn as nn

def convert_torch(img: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(img).permute(2, 0, 1)  # Change from HWC to CHW format
    tensor = tensor.float() / 255.0  # Normalize to [0, 1]
    return tensor

def convert_numpy(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.permute(1, 2, 0)  # Change from CHW to HWC format
    img = (tensor.numpy() * 255).astype(np.uint8)  # Convert to uint8
    return img

def resize_image(image, size: tuple) -> torch.Tensor:
    def resize_PIL(image: Image.Image, size: tuple) -> Image.Image:
        image = image.convert("RGB")
        return image.resize(size, Image.LANCZOS)
    
    def resize_cv2(image: np.ndarray, size: tuple) -> np.ndarray:
        return cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)
    
    if isinstance(image, Image.Image):
        resized = np.array(resize_PIL(image, size))
    elif isinstance(image, np.ndarray):
        resized = resize_cv2(image, size)
    else:
        raise TypeError("Unsupported image type. Expected PIL Image or numpy array.")
    
    return convert_torch(resized)