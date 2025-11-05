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

import os
import psutil
import time
import threading
import datetime
import torch
import mlflow
import pandas as pd
import shutil
import yaml
from ultralytics import settings

import config.mlflow_config as mlflow_config
import src.utils.mlflow_utils as mlflow_utils

settings.update({"mlflow": False})

mlflow_utils.clear_tmp()

results = {}

stop_event = threading.Event()

def monitor_usage(interval=1):
    average_ram, peak_ram, average_vram, peak_vram, total_time, average_cores, peak_cores = 0, 0, 0, 0, 0, 0, 0

    pid = os.getpid()
    process = psutil.Process(pid)

    start_time = time.time()
    while not stop_event.is_set():
        ram_usage = process.memory_info().rss / (1024**3)

        peak_ram = ram_usage if peak_ram < ram_usage else peak_ram
        average_ram += ram_usage
        
        vram_usage = torch.cuda.memory_reserved(0) / (1024**3)

        average_vram+=vram_usage
        peak_vram = vram_usage if peak_vram < vram_usage else peak_vram

        cores_usage = psutil.cpu_percent(interval=None, percpu=True)

        cores = sum(1 for core in cores_usage if core > 0)
        average_cores += cores
        peak_cores = cores if peak_cores < cores else peak_cores
        
        time.sleep(interval)

    total_time = time.time() - start_time
    total_time = int(total_time)

    results["mean_RAM"] = round(average_ram/total_time,2)
    results["peak_RAM"] = round(peak_ram,2)
    results["mean_VRAM"] = round(average_vram/total_time,2)
    results["peak_VRAM"] = round(peak_vram,2)
    results["mean_COREs"] = round(average_cores/total_time,1)
    results["peak_COREs"] = int(peak_cores)
    results["time"] = total_time

def read_results():
    df_results = pd.read_csv("models/object_detector/onnx_yolo_obj_det/results.csv")
    df_results = df_results[df_results["    metrics/mAP50-95(B)"] == df_results["    metrics/mAP50-95(B)"].max()].head(1)
    dict_res = {}
    for col in df_results.columns:
        col_name = col.strip().split("(")
        col_name=col_name[0]
        dict_res[col_name] = df_results[col].values[0]

    return dict_res


def run_training(dataset:DatasetFactory):
    run_name = "CAPSULIA_TRAINING_COUNT_PEOPLE_YOLOV8"
    model_path = "models/object_detector/onnx_yolo_obj_det/"

    experiment=mlflow_utils.set_mlflow_config(mlflow_config.PREFIX_EXP_NAME)
    with (mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name) as _):
        thread = threading.Thread(target=monitor_usage, daemon=True)
        thread.start()

        model.train(dataset)

        stop_event.set()
        thread.join()

        with open(os.path.join(model_path,"args.yaml")) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            mlflow.log_params(params)
        model_uri = mlflow.get_artifact_uri("")

        mlflow.set_tag(value="YOLOV8", key="MODEL")

        mlflow.set_tag("mlflow.runName", run_name)

        results_metrics = read_results()
        mlflow.log_metrics(results_metrics)
        mlflow.log_metrics(results)
        
    objects_to_upload = ["results.csv","results.png","args.yaml","confusion_matrix_normalized.png"]
    for object in objects_to_upload:
        shutil.copy(os.path.join(model_path,object),os.path.join(mlflow_config.TMP_PATH,object))

    mlflow_utils.upload_artifacts(model_uri=model_uri)
    mlflow_utils.clear_tmp()

if __name__ == '__main__':
    cfg = RunParser("config/run.cfg")
    img_size = int(cfg.img_size)
    img_size = (img_size, img_size)
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])

    dataset = DatasetFactory(cfg=cfg,
                             csv_file=IMAGES_LABELS_PATH,
                             img_dir=IMAGES_PATH,
                             transform=transform)

    model = ModelFactory(cfg=cfg,
                         label_encoding=dataset.label_encoding)
    model.train(dataset)

