#!/usr/bin/env python3
# coding: utf-8
# Copyright (C) Solver Intelligent Analytics -All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by SolverIA <info@iasolver.com>, febrero 2024
#
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

class ImageClassificationDataset(Dataset):
    """A custom dataset class for loading images based on labels from a CSV file.

        Attributes:
            labels_frame (DataFrame): DataFrame containing image names and labels.
            img_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on an image sample.
        """

    def __init__(self, label_df, num_classes, label_encoding, img_dir, transform=None, *args, **kwargs):
        """
        Args:
            csv_file (str): Path to the CSV file with annotations.
            img_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.label_df = label_df
        self.img_dir = img_dir
        self.transform = transform
        self.num_classes = num_classes   
        self.label_encoding = label_encoding
         
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
    def __init__(self,csv_file,img_dir, transform,stratify_ratio = 0.2, *args, **kwargs):
        self.label_df = pd.read_csv(os.path.join(csv_file,"labels.csv"))
        self.num_classes,self.label_encoding = self.get_classes()
        self.train_images = int((1 - float(stratify_ratio))*len(self.label_df))
        self.train = ImageClassificationDataset(self.label_df.iloc[:self.train_images].reset_index(drop=True), self.num_classes, self.label_encoding, img_dir,transform)
        self.val = ImageClassificationDataset(self.label_df.iloc[self.train_images:].reset_index(drop=True), self.num_classes, self.label_encoding, img_dir,transform)

    def get_classes(self):
        if(len(self.label_df.columns)>2):
            self.label_df = self.transform_df()
        label_list=[]
        for label in self.label_df["LABELS"]:
            labels = label.split(",")
            for i in labels:
                if i.strip() != "":
                    label_list.append(i.strip())

        labels = list(set(label_list))
        encoding={}
        for i in range(len(labels)):
            encoding[labels[i]] = i
        return len(labels), encoding

    def transform_df(self):
        df = self.label_df.copy()
        label_cols = [col for col in df.columns if col != "IMAGE_NAME"]
        df["LABELS"] = df.apply(lambda row: ','.join([col for col in label_cols if row[col]==1]), axis=1)
        return df[["IMAGE_NAME","LABELS"]].sample(frac=1, replace = False, random_state = 23)   
