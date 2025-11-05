#!/usr/bin/env python3
# coding: utf-8
# Copyright (C) Solver Intelligent Analytics -All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by SolverIA <info@iasolver.com>, febrero 2024
#

import random
import configparser
import torch
import numpy as np

from config.config import YOLO_MODEL, DETR_MODEL, TIMM_MODEL, DEFAULT_MODEL


def read(cfg, section, value, default_value, is_boolean=False):
    try:
        if not is_boolean:
            value = cfg[section][value]
        else:
            value = cfg[section].getboolean(value)
    except Exception:
        value = default_value
    return value


class RunParser:
    def __init__(self, fname):
        config = configparser.ConfigParser()
        config.read(fname)
        self.type = read(config, "Task", "Type", "classification")
        self.img_size = read(config, "Images", "Size", "256")
        self.transform = read(config, "Images", "DataAug", "gray")
        self.model = read(config, "Model", "Name", "densenet121")
        self.stratify = read(config, "Preprocess", "Stratify", "yes", is_boolean=True)
        self.multilabel = read(config, "Preprocess", "MultiLabel", "no", is_boolean=True)
        self.pretrained = read(config, "ModelParams", "Pretrained", "yes", is_boolean=True)
        self.overwrite = read(config, "ModelParams", "Overwrite", "yes", is_boolean=True)
        self.checkpoint = read(config, "Model", "Checkpoint", None)
        self.batch = read(config, "LearningParams", "Batch", "32")
        self.epoch = read(config, "LearningParams", "Epoch", "25")
        self.eval_partition = read(config, "LearningParams", "EvalPartition", "0.2")
        self.optim = read(config, "LearningParams", "Optim", "Adam")
        self.lr = read(config, "LearningParams", "LearningRate", "0.001")
        self.scheduler = read(config, "LearningParams", "Scheduler", "Cosine")
        self.patience = read(config, "LearningParams", "Patience", None)


def load_devices():
    """
    Determines the best available device for computation and the
    number of GPUs available.

    Returns:
        Tuple[torch.device, int]: A tuple containing the selected
        computation device (either a CUDA device or CPU) and
        the total number of GPUs available. If CUDA is not available,
        the number of GPUs will be 0.
    """
    num_gpus = 0
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        num_gpus = torch.cuda.device_count()
    else:
        device = torch.device("cpu")
    return device, num_gpus


def seed(seed=16):
    """
    Sets the seed for generating random numbers
    to ensure reproducibility.
    Args:
        seed (int): The seed value to use for random
         number generators across various libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def base_model(model_name: str):
    if YOLO_MODEL in model_name.lower():
        return YOLO_MODEL
    if DETR_MODEL in model_name.lower():
        return DETR_MODEL
    return DEFAULT_MODEL

