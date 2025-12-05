#!/usr/bin/env python3
# coding: utf-8
# Copyright (C) Solver Intelligent Analytics -All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by SolverIA <info@iasolver.com>, febrero 2024
#
import yaml
import json
import argparse
import os
import sys
import torch

from src.utils.utils import RunParser
from src.inference.inference_classification import Timm_Inference
from src.inference.inference_detection import YOLOv8_Inference

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run image classification inference.")
    parser.add_argument('--img_folder', type=str, help='Path to the folder containing images.',required=True)
    parser.add_argument('--model_path', type=str, default='models/classificator/model.onnx',
                        help='Path to the model checkpoint.')
    parser.add_argument('--task', type=str, default='c',
                        help='Classify or detect. Write c to classify and d to detect.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
 
    if not os.path.exists("config/run.cfg"):
        print(f"The config file does not exist in [config/run.cfg]")
        sys.exit(1)
    
    cfg = RunParser("config/run.cfg")
    
    if not os.path.exists(args.img_folder):
        print(f"The specified image folder does not exist: {args.img_folder}")
        sys.exit(1)

    classes = range(80)
    if os.path.exists("data/label2id.json"):
        with open("data/label2id.json") as f:
            labels = json.load(f)
            classes = [k for k,v in labels.items()]
    
    if args.task=='c':       
        inference = Timm_Inference(args.model_path, args.img_folder, int(cfg.img_size), cfg.multilabel, classes)
    else:
        inference = YOLOv8_Inference(args.model_path, args.img_folder, int(cfg.img_size), 0.5, 0.5, classes)
   
    inference.main()
