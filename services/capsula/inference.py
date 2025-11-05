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
from torch.utils.data import DataLoader

from src.inference.inference_classification import InferenceDataset, do_inference
from src.inference.inference_detection import YOLOv8_Inference
from src.models.classes.classification_timm_model import ClassificationTimmModel

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run image classification inference.")
    parser.add_argument('--img_folder', type=str, help='Path to the folder containing images.',required=True)
    parser.add_argument('--model_path', type=str, default='models/classificator/last_model',
                        help='Path to the model checkpoint.')
    parser.add_argument('--task', type=str, default='c',
                        help='Classify or detect. Write c to classify and d to detect.')
    return parser.parse_args()


def classification_main(args):
    if not os.path.exists(args.img_folder):
        print(f"The specified image folder does not exist: {args.img_folder}")
        sys.exit(1)

    files_path = os.listdir(args.img_folder)
    imgs_path = [os.path.join(args.img_folder, file) for file in files_path if
                 file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    checkpoint = torch.load(args.model_path)
    
    img_size = int(checkpoint['img_size'])
    img_size = (img_size, img_size)

    label_encoding = checkpoint['class_encoding']

    num_classes = len(label_encoding.keys())
    model = ClassificationTimmModel(cfg=None,num_classes=num_classes, label_encoding=label_encoding,
                                checkpoint=checkpoint,
                                is_inference=True)
    id_to_label = {value: key for key, value in label_encoding.items()}

    dataset = InferenceDataset(image_paths=imgs_path, img_size=img_size)
    dataloader = DataLoader(dataset, batch_size=model.batch_size, shuffle=False)

    infs, probs = do_inference(model, dataloader, id_to_label)
    for i in range(len(imgs_path)):
        for j in range(len(infs[i])):
            cls = infs[i][j]
            prob = round(probs[i][j].item(), 4)
            img_name = imgs_path[i]
            print(f"Para la imagen {img_name} el modelo ha predicho la clase {cls} con una confianza de {prob}.")

def detection_main(args):
    if not os.path.exists(args.img_folder):
        print(f"The specified image folder does not exist: {args.img_folder}")
        sys.exit(1)

    files_path = os.listdir(args.img_folder)
    imgs_path = [file for file in files_path if
                 file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    with open(os.path.join(args.model_path,"args.yaml")) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

    classes = range(80)
    if os.path.exists("data/label2id.json"):
        with open("data/label2id.json") as f:
            labels = json.load(f)
            classes = [v for k,v in labels.items()]
    print(classes)
    detection = YOLOv8_Inference(os.path.join(args.model_path,"weights/best.onnx"), args.img_folder, imgs_path, 0.5, params['iou'], params["imgsz"], classes)
    detection.main()


if __name__ == '__main__':
    args = parse_arguments()
    classification_main(args) if args.task=='c' else detection_main(args)
