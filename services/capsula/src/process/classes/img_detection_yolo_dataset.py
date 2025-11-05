#!/usr/bin/env python3
# coding: utf-8
# Copyright (C) Solver Intelligent Analytics -All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by SolverIA <info@iasolver.com>, febrero 2024
#

import os
from config.config import DATA_BASE_PATH, DATA_FINAL_PATH, OBJ_DETECTION_YAML_PATH
from src.process.data_process import is_grayscale_dataset
from src.utils.yolo_utils import create_dataset_yaml, get_label2id, create_stratified_yolo_dataset

class ObjectDetectionYoloDataset:
    """
    Prepare data to be used in a YOLO detection Model.

    Attributes:
        img_dir (str): Directory with all the images.
    """

    def __init__(self, img_dir, stratify=True, stratify_ratio=0.2,*args, **kwargs):
        self._is_grayscale = is_grayscale_dataset(img_dir)
        self._label_encoding = get_label2id(DATA_BASE_PATH)
        self._num_classes = len(self.label_encoding.keys())
        self.class_names = sorted(self.label_encoding, key=self.label_encoding.get)
        self.__create_dataset_yaml(stratified=stratify)
        if stratify:
            create_stratified_yolo_dataset(base_dir=DATA_BASE_PATH, 
                                           stratify_ratio=stratify_ratio,
                                           label2id=self._label_encoding)


    @property
    def label_encoding(self):
        return self._label_encoding
    
    @label_encoding.setter
    def label_encoding(self, value: dict):
        if not isinstance(value, dict):
            raise TypeError ("Label encoding should be a dictionary")
        self._label_encoding = value
    
    @property
    def num_classes(self):
        return self._num_classes
    
    @num_classes.setter
    def num_classes(self, value: int):
        if not isinstance(value, int):
            raise TypeError ("The number of classes should be an integer")
        self._num_classes = value
    
    @property
    def is_grayscale(self):
        return self._is_grayscale
    
    @is_grayscale.setter
    def is_grayscale(self, value: bool):
        self._is_grayscale = value

    
    
    def __create_dataset_yaml(self, stratified: bool):
        path = DATA_FINAL_PATH if stratified else DATA_BASE_PATH
        absolute_path = os.path.abspath(path)
        num_classes = self.num_classes
        class_names = self.class_names
        create_dataset_yaml(absolute_path, 
                            num_classes=num_classes,
                            class_names=class_names,
                            yaml_file_path=OBJ_DETECTION_YAML_PATH)  
    