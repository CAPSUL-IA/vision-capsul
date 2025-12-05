#!/usr/bin/env python3
# coding: utf-8
# Copyright (C) Solver Intelligent Analytics -All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by SolverIA <info@iasolver.com>, febrero 2024
#
import os
import json
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

class ImageClassificationDataset(DataLoader):
    """A custom dataset class for loading images based on labels from a CSV file.

        Attributes:
            labels_frame (DataFrame): DataFrame containing image names and labels.
            img_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on an image sample.
        """

    def __init__(self, label_df, num_classes, label_encoding, img_dir, img_size, split, *args, **kwargs):
        """
        Args:
            csv_file (str): Path to the CSV file with annotations.
            img_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.label_df = label_df
        self.img_dir = img_dir
        self.num_classes = num_classes   
        self.label_encoding = label_encoding
        self.img_size = img_size
        self.split = split

        self.transform = self.get_transform()
        
    def get_transform(self):
        if self.split == "train":
            return transforms.Compose([
                transforms.RandAugment(num_ops=2, magnitude=9),  
                transforms.Resize((self.img_size, self.img_size)),        
                transforms.ToTensor(),                         
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),        
                transforms.ToTensor(),                         
            ])

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.label_df)

    def __getitem__(self, idx):
        """
        Retrieves the image and its labels at the specified
         index, applying any transformations.
        Args:
            idx (int): The index of the item.
        Returns:
            tuple: A tuple containing the transformed image
             and its corresponding label tensor.
        """
        if idx in self.label_df.index:
            img_path = os.path.join(self.img_dir, self.label_df.loc[idx, "IMAGE_NAME"])
            str_labels = self.label_df.loc[idx, "LABELS"].split(',')
            labels = torch.zeros(len(self.label_encoding))
            id_labels = [self.label_encoding[label.strip()] for label in str_labels if label.strip() != '']
            labels[id_labels] = 1
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, labels
        else:
            raise IndexError
        
class ClassificationDataset:
    def __init__(self, labels_path, img_dir, img_size, frac=1.0, stratify_ratio=0.2, *args, **kwargs):
        """
        labels_path: directory where labels.csv is located
        img_dir: directory with images
        transform: image transformations
        frac: fraction of data to use
        stratify_ratio: validation split ratio
        target_col: target column in the CSV
        """
        self.labels_path = labels_path
        self.img_dir = img_dir
        self.img_size = int(img_size)
        self.frac = float(frac)
        self.stratify_ratio = float(stratify_ratio)

        self.label_df = self.read_and_frac_labels(self.labels_path, self.frac)

        self.num_classes, self.label_encoding = self.get_classes()

        self.create_train_val_datasets()

    def read_and_frac_labels(self, path, frac):
        df = pd.read_csv(os.path.join(path, "labels.csv"))
        
        # Transform if there are multiple label columns
        if len(df.columns) > 2:
            df = self.transform_df(df)
        
        # Sample n_take per class and shuffle
        df = df.groupby("LABELS", group_keys=False).apply(
            lambda x: x.sample(n=max(1, int(len(x)*frac)), random_state=42)
        )

        return df.sample(frac=1, random_state=42).reset_index(drop=True)

    def transform_df(self, df):
        """Transform multiple label columns into a single 'LABELS' column"""
        label_cols = [col for col in df.columns if col != "IMAGE_NAME"]
        df["LABELS"] = df.apply(lambda row: ','.join([col for col in label_cols if row[col] == 1]), axis=1)
        return df[["IMAGE_NAME", "LABELS"]]

    def get_classes(self):
        """Generate class encoding"""
        label_list = []
        for label in self.label_df["LABELS"]:
            labels = label.split(",")
            label_list.extend([l.strip() for l in labels if l.strip() != ""])

        unique_labels = sorted(set(label_list))
        encoding = {label: idx for idx, label in enumerate(unique_labels)}
       
        with open("data/label2id.json", "w", encoding="utf-8") as f:
            json.dump(encoding, f, ensure_ascii=False, indent=4)

        return len(unique_labels), encoding

    def create_train_val_datasets(self):
        """Split the dataset into train and validation sets"""
        n_train = int((1 - self.stratify_ratio) * len(self.label_df))
        train_df = self.label_df.iloc[:n_train].reset_index(drop=True)
        val_df = self.label_df.iloc[n_train:].reset_index(drop=True)

        self.train = ImageClassificationDataset(train_df, self.num_classes, self.label_encoding, self.img_dir, self.img_size, "train")
        self.val = ImageClassificationDataset(val_df, self.num_classes, self.label_encoding, self.img_dir, self.img_size, "val")
