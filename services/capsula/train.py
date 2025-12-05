#!/usr/bin/env python3
# coding: utf-8
# Copyright (C) Solver Intelligent Analytics -All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by SolverIA <info@iasolver.com>, febrero 2024
#
from torchvision import transforms
from src.utils.utils import RunParser
from config.config import IMAGES_PATH, IMAGES_LABELS_PATH
from src.models.classes.model_factory import ModelFactory
from src.process.classes.dataset_factory import DatasetFactory

from ultralytics import settings

settings.update({"mlflow": False})


if __name__ == '__main__':
    cfg = RunParser("config/run.cfg")

    dataset = DatasetFactory(cfg=cfg,
                             labels_path=IMAGES_LABELS_PATH,
                             img_dir=IMAGES_PATH,
                             img_size = cfg.img_size,
                             frac = cfg.fraction)

    model = ModelFactory(cfg=cfg,
                         label_encoding=dataset.label_encoding)
    model.train(dataset)

