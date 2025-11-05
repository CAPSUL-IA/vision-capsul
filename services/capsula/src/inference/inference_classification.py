#!/usr/bin/env python3
# coding: utf-8
# Copyright (C) Solver Intelligent Analytics -All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by SolverIA <info@iasolver.com>, febrero 2024
#
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import datasets, transforms


class InferenceDataset(Dataset):
    """
    Custom dataset class to load images from a list of file paths.
    Attributes:
        image_paths (list of str): List of image file paths.
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, image_list=None, image_paths=None, img_size: tuple = (28, 28)):
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
        self.image_paths = image_paths
        self.image_list = image_list
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        """Returns the number of images in the dataset."""
        if self.image_paths is not None:
            return len(self.image_paths)
        else:
            return len(self.image_list)

    def __getitem__(self, idx):
        """
        Fetches the image at the given index in the dataset.

        Args:
            idx (int): Index of the image to fetch.

        Returns:
            torch.Tensor: Transformed image.
        """
        if self.image_paths is not None:
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
        else:
            image = self.image_list[idx].convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def process_multiclass_output(outputs, label_encoding):
    """
    Processes the output of the model for a multiclass
    classification task.
    Args:
        outputs (torch.Tensor): The raw output from the model
         for a batch of images.
        label_encoding (dict): A dictionary mapping numerical
         labels to their corresponding string representation.
    Returns:
        tuple: A tuple containing a list of labels for each image
         and a tensor of probabilities associated with the
         predicted label for each image.
    """
    probs = torch.nn.functional.softmax(outputs, dim=1)
    image_probs_list, preds = torch.max(probs, dim=1)
    image_labels = [label_encoding[pred.item()] for pred in preds]
    return image_labels, image_probs_list

def process_multilabel_output(outputs, label_encoding):
    """
    Processes the output of the model for a multiclass
    classification task.
    Args:
        outputs (torch.Tensor): The raw output from the model
         for a batch of images.
        label_encoding (dict): A dictionary mapping numerical
         labels to their corresponding string representation.
    Returns:
        tuple: A tuple containing a list of labels for each image
         and a tensor of probabilities associated with the
         predicted label for each image.
    """
    image_labels, image_probs_list = [], []
    probs = torch.nn.functional.sigmoid(outputs)
    for probab in probs:
        index_preds = [i for i,prob in enumerate(probab) if prob>0.5]
        image_labels.append([label_encoding[i] for i in index_preds])
        image_probs_list.append([probab[i] for i in index_preds])

    return image_labels, image_probs_list


def do_inference(model, dataloader, label_encoding):
    """
    Performs inference on the given dataloader with the model.
    Args:
        model (torch.nn.Module): Trained model for inference.
        dataloader (torch.utils.data.DataLoader): DataLoader
         for batching the inference.
        label_encoding (dict): Dictionary mapping numerical labels
         to their real name
    Returns:
        final_preds (list): List of predicted classes for all instances.
        final_probs (list): List of probabilities associated with the
         predicted class for all instances.
    """
    final_preds = []
    final_probs = []
    for images in dataloader:
        images = images.to(model.device)
        with torch.no_grad():
            outputs = model.model(images)
            if model.multilabel:
                labels, probs = process_multilabel_output(outputs,
                                                      label_encoding)
            else:
                labels, probs = process_multiclass_output(outputs,
                                                      label_encoding)
            final_preds.extend(labels)
            final_probs.extend(probs)
    return final_preds, final_probs
