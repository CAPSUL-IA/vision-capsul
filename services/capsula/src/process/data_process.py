import os
import numpy as np
from PIL import Image
from config.config import THR_RB, THR_RG, THR_BG, THR_PIXELS

def is_grayscale_img(image_path: str) -> bool:
    """
    Function to know if an image is in grayscale or color.
    Args:
        image_path: The path to the image file.
    Returns:
        bool: True if the image is greyscale, False otherwise.
    """
    pil_image = Image.open(image_path)
    pil_image = pil_image.convert('RGB')
    img = np.array(pil_image)
    total_pixels = img.size
    if img.shape[2] == 1:
        return True
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    if (b == g).all() and (b == r).all():
        return True
    else:
        r_g = abs(r.astype(np.float32) - g.astype(np.float32))
        r_b = abs(r.astype(np.float32) - b.astype(np.float32))
        b_g = abs(b.astype(np.float32) - g.astype(np.float32))
        count_rb = np.where(r_b > THR_RB, 1, 0)
        count_rg = np.where(r_g > THR_RG, 1, 0)
        count_bg = np.where(b_g > THR_BG, 1, 0)
        sum_counts = count_rb + count_rg + count_bg
        pixel_color = np.count_nonzero(np.where(sum_counts > 0, 1, 0))
        if pixel_color < THR_PIXELS * total_pixels:
            return True
    return False


def is_grayscale_dataset(images_path: str):
    """
    Function to check if the images are in grayscale
    transform they to RGB.
    Args:
        images_path: path where images are stored.
    Returns:
    """
    any_grayscale = False
    for img in os.listdir(images_path):
        img_path = os.path.join(images_path, img)
        if os.path.isfile(img_path):
            try:
                if is_grayscale_img(img_path):
                    any_grayscale = True
                img = Image.open(img_path)
                img_rgb = img.convert('RGB')
                img_rgb.save(img_path)
            except Exception:
                pass
    return any_grayscale


def build_data_aug_dic(is_bn: bool) -> dict:
    """
    Function to build a dictionary with data augmentation
    for Yolo training YAML depending on if images are in
    grayscale or color.
    Args:
        is_bn: Bool to indicate if images are grayscale.
    Returns:
        Dictionary with data augmentation parameters and values.
    """
    if not is_bn:
        augmentations = {
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
            'flipud': 0.0,
            'fliplr': 0.0,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.15,
            'shear': 0.0,
            'perspective': 0.0,
            'hsv_h': 0.015,
            'hsv_s': 0.2,
            'hsv_v': 0.2,
            'erasing': 0.05
        }
    else:
        augmentations = {
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
            'flipud': 0.0,
            'fliplr': 0.0,
            'degrees': 0.1,
            'scale': 0.15,
            'shear': 0.1,
            'perspective': 0.1,
            'hsv_h': 0,
            'hsv_s': 0,
            'hsv_v': 0.2,
            'erasing': 0.05
        }
    return augmentations