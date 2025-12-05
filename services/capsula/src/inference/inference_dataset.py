import os
import cv2
import numpy as np

from torch.utils.data import DataLoader


class InferenceDataset(DataLoader):
    """
    Custom dataset class to load images from a list of file paths.
    Attributes:
        image_paths (list of str): List of image file paths.
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, image_path=None, img_size: tuple = (28, 28)):
        """
        Initializes the dataset with image paths and an optional transform.

        Args:
            image_list (list of Pil Image): List of images with PIL format. It
            is used for inference through Api. If this is None, image_paths
            should not be None.
            image_paths (list of str): List of image file paths. It is used for
            inference of local data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.path = image_path
        self.imgs = os.listdir(self.path)
        self.img_size = img_size 

    def __len__(self):
        """Returns the number of images in the dataset."""
        return len(self.imgs)

    def __getitem__(self, idx):
        """
        Fetches the image at the given index in the dataset.

        Args:
            idx (int): Index of the image to fetch.

        Returns:
            torch.Tensor: Transformed image.
        """
        image_path = self.imgs[idx]

        image = cv2.imread(os.path.join(self.path,image_path))
        image = cv2.resize(image, self.img_size)

        image = np.array(image) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0).astype(np.float32)

        return image
