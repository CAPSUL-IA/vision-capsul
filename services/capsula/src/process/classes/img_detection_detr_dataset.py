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
from config.config import DATA_BASE_PATH
from src.utils.yolo_utils import get_label2id

import albumentations as A
import numpy as np

class ObjectDetectionDETRDataset(Dataset):
    """
    Prepare data to be used in a YOLO detection Model.

    Attributes:
        img_dir (str): Directory with all the images.
    """
    def __init__(self, task, label_encoding, image_processor=None, transform=None):
        self.task = task
        self.label_encoding = label_encoding
        self.image_path = os.path.join(DATA_BASE_PATH,"images",task)
        self.label_path = os.path.join(DATA_BASE_PATH,"labels",task)
        self.dataset = self.load_dataset()
        self._num_classes = len(self.label_encoding.keys())
        self.class_names = sorted(self.label_encoding, key=self.label_encoding.get)
        self.image_processor = image_processor
        self.transform = self.create_transform()

    def create_transform(self):
        if self.task=="train":
            return A.Compose(
                [
                    A.Perspective(p=0.1),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                    A.HueSaturationValue(p=0.1),
               ],
                bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=25, min_width=1, min_height=1),
            )
        else: 
            return A.Compose(
                [A.NoOp()],
                bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=1, min_width=1, min_height=1),
            )

    def load_dataset(self):
        dataset = []
        for id in os.listdir(self.image_path):
            name, _ = os.path.splitext(id)
            categories, bboxes = self.get_labels(name)
            dataset.append(
                    {
                        "image_path": os.path.join(self.image_path,id),
                        "bboxes":bboxes,
                        "categories":categories
                    }
                    )
        return dataset


    @staticmethod
    def annotations_to_coco(image_id, categories, boxes):
        annotations = []
        for category, bbox in zip(categories, boxes):
            formatted_annotation = {
                "image_id": image_id,
                "category_id": int(category),
                "bbox": list(bbox),
                "iscrowd": 0,
                "area": bbox[2] * bbox[3],
            }
            annotations.append(formatted_annotation)

        return {
            "image_id": image_id,
            "annotations": annotations,
        }
    def get_labels(self, label):
        categories, bboxes = [], []
        with open(os.path.join(self.label_path,f"{label}.txt"),"r",encoding="utf-8") as labels:
            for l in labels.readlines():
                l= l.strip().split(" ")
                categories.append(int(l[0]))
                bboxes.append([float(b) for b in l[1:]])
                
        return categories, bboxes
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        image_path,bboxes,categories = data["image_path"],data["bboxes"],data["categories"]
        
        image = np.array(Image.open(image_path).convert("RGB"))
        height, width = image.shape[:2]
    
        coco_bboxes = []
        for bbox in bboxes:
            x, y, w, h = bbox
            x_min = (x - w/2) * width
            y_min = (y - h/2) * height
            w_pixels = w * width
            h_pixels = h * height
            coco_bboxes.append([x_min, y_min, w_pixels, h_pixels])

        transformed = self.transform(image=image, bboxes=coco_bboxes, category=categories)
        image = transformed["image"]
        bboxes = transformed["bboxes"]
        categories = transformed["category"]

        formatted_annotations = self.annotations_to_coco(idx, categories, bboxes)
        
        result = self.image_processor(
            images=image, annotations=formatted_annotations, return_tensors="pt"
        )

        result_values = {
                    "pixel_values": result["pixel_values"].squeeze(0), "labels": result["labels"][0]
                }
        if "pixel_mask" in result:
            result_values["pixel_mask"] = result["pixel_mask"].squeeze(0)
        return result_values

class DETRDataset:
    def __init__(self,*args, **kwargs):
        self.label_encoding = get_label2id(DATA_BASE_PATH)
        self.train = ObjectDetectionDETRDataset("train", self.label_encoding)
        self.val = ObjectDetectionDETRDataset("val", self.label_encoding)
