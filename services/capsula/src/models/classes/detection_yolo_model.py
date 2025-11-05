#!/usr/bin/env python3
# coding: utf-8
# Copyright (C) Solver Intelligent Analytics -All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by SolverIA <info@iasolver.com>, febrero 2024
#
from ultralytics import YOLO, RTDETR
from config.config import OBJ_DETECTION_YAML_PATH, OBJ_DETECTION_MODEL_DIR
from src.models.classes.onnx_yolo_model import OnnxYoloModel
from src.process.data_process import build_data_aug_dic
from src.models.interfaces.model import IModel
from src.utils.utils import RunParser, load_devices


class DetectionYoloModel(IModel):
    """
    A class representing an object detection model.

    This class encapsulates the model configuration, training,
    and inference processes, handling the setup and execution of
    model training and predictions based on the
    provided configurations.

    Attributes:
        model_name (str): The name of the model, derived from
          the checkpoint path if provided, otherwise from cfg.
        device (str): The device (CPU/GPU) on which the model is loaded.
        num_gpus (int): The number of GPUs available for training.
        model (object): The loaded model object, ready for training or inference.
        learning_rate (float): The learning rate for model training.
        optimizer (str): The optimizer used for model training.
        epochs (int): The number of epochs to train the model.
        batch_size (int): The batch size used during training.
        img_size (int): The input image size for the model.
        patience (int): The patience for early stopping.
        label_encoding (dict): A dictionary mapping class identifiers to labels.
    """

    def __init__(self, cfg: RunParser,
                 checkpoint_path: str = None,
                 label_encoding: dict = None,
                 *args,
                 **kwargs):
        """
        Initializes the object detection model with configuration and optional
        parameters for data augmentation and checkpoint loading.
        
        Args:
            cfg (RunParser): The configuration settings for the model.
            checkpoint_path (str, optional): Path to a pre-trained model checkpoint.
            label_encoding (dict): A dictionary mapping class identifiers to labels.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.model_name = cfg.model
        self.checkpoint = checkpoint_path
        self.label_encoding = label_encoding
        self.img_size = int(cfg.img_size)
        self.batch_size = int(cfg.batch)
        self.device, self.num_gpus = load_devices()
        self.model = self.load_model()
        self.learning_rate = float(cfg.lr)
        self.optimizer = cfg.optim
        self.epochs = int(cfg.epoch)
        self.patience = int(cfg.patience) if cfg.patience is not None else None
        self.is_grayscale = None
        self.data_augmentation = None
        self.overwrite = cfg.overwrite

    def load_model(self):
        """
        Loads the model based on the configuration settings.

        Returns:
            The loaded model object, ready for training or inference.
        """
        if self.checkpoint is not None:
            model_path = self.checkpoint
        else:
            model_path = self.model_name

        if "onnx" in model_path:
            model = OnnxYoloModel(model_path, self.label_encoding,
                              self.img_size, self.batch_size,
                              self.model_name)
        elif "detr" in self.model_name:
            model = RTDETR(model_path).to(self.device)
        else:
            model = YOLO(model_path, task='detect').to(self.device)
        return model

    def train(self, dataset):
        """
        Trains the model using the specified configurations and data augmentation
        parameters.
        """
        if self.is_grayscale is None or self.data_augmentation is None:
            self.is_grayscale = dataset.is_grayscale
            self.data_augmentation = build_data_aug_dic(self.is_grayscale)

        devices = [i for i in range(self.num_gpus)] if self.num_gpus > 0 else None

        name = "onnx_yolo_obj_det"
        self.model.train(data=OBJ_DETECTION_YAML_PATH,
                         imgsz=self.img_size,
                         batch=self.batch_size,
                         epochs=self.epochs,
                         lr0=self.learning_rate,
                         optimizer=self.optimizer,
                         patience=self.patience,
                         project=OBJ_DETECTION_MODEL_DIR,
                         exist_ok=self.overwrite,
                         name=name,
                         device=devices,
                         iou=0.5,
                         **self.data_augmentation)
        
        self.model.export(format='onnx', imgsz=self.img_size, dynamic=True)
    

    def inference(self, data: list):
        """
        Runs inference on the provided data and post-processes the results.

        Args:
            data (list): The data to run inference on.

        Returns:
            Dict with the processed inference results.
        """
        # if "onnx" in self.checkpoint:
        #     boxes, scores, classes = self.model(data)
        #     results = postprocess_onnx_inference(boxes, scores, classes, self.id_label)
        # else:
        #     inference = self.model(data, imgsz=self.img_size,
        #                            stream=False,
        #                            verbose=False)
        #     results = postprocess_inference(inference, self.id_label)
        # return results
        pass
    
