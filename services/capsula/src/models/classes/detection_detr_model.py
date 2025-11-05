#!/usr/bin/env python3
# coding: utf-8
# Copyright (C) Solver Intelligent Analytics -All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by SolverIA <info@iasolver.com>, agosto 2025
#
import pandas as pd
import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, TaskType, get_peft_model

from config.config import OBJ_DETECTION_YAML_PATH, OBJ_DETECTION_MODEL_DIR

from src.models.interfaces.model import IModel
from src.utils.utils import RunParser, load_devices
from src.process.classes.img_detection_detr_dataset import ObjectDetectionDETRDataset
from src.utils.MAPEvaluator import MAPEvaluator

class DetectionDetrModel(IModel):
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
        self.label_encoding = label_encoding
        self.label2id = {self.label_encoding[key]:key for key in self.label_encoding.keys()}
        self.img_size = int(cfg.img_size)
        self.batch_size = int(cfg.batch)
        self.device, self.num_gpus = load_devices()
        self.model = self.load_model()
        self.processor = self.load_processor()
        self.learning_rate = float(cfg.lr)
        self.epochs = int(cfg.epoch)
        self.patience = int(cfg.patience) if cfg.patience is not None else None
        self.is_grayscale = None
        self.data_augmentation = None
        self.overwrite = cfg.overwrite
        self.eval_compute_metrics_fn = MAPEvaluator(image_processor=self.processor, threshold=0.01, id2label=self.label2id)

    def load_model(self):
        model = AutoModelForObjectDetection.from_pretrained(
                self.model_name,
                id2label=self.label_encoding,
                label2id=self.label2id,
                num_labels=len(self.label_encoding),
                anchor_image_size=None,
                ignore_mismatched_sizes=True,
        )

        model = model.to(self.device)
        
        return model

    def load_processor(self):
        processor = AutoImageProcessor.from_pretrained(
            self.model_name,
            use_fast=True,
            do_resize=True,
            size={"height": self.img_size, "width": self.img_size},
            do_pad=True,
            pad_size={"height": self.img_size, "width": self.img_size}
        )

        return processor
    
    def collate_fn(self,batch):
        data = {}
        data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
        data["labels"] = [x["labels"] for x in batch]
        if "pixel_mask" in batch[0]:
            data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
        return data

    def train(self, dataset):
        """
        Trains the model using the specified configurations and data augmentation
        parameters.
        """
        
        dataset.train.image_processor = self.processor
        dataset.val.image_processor = self.processor
        
        training_args = TrainingArguments(
            output_dir=f"models/rt-detr",
            num_train_epochs=self.epochs,
            max_grad_norm=0.1,
            learning_rate=self.learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size, 
            dataloader_num_workers=4,
            eval_accumulation_steps=None,
            metric_for_best_model="eval_map",
            greater_is_better=True,
            load_best_model_at_end=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            remove_unused_columns=False,
            eval_do_concat_batches=False,
            logging_steps=100
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset.train,
            eval_dataset=dataset.val,
            tokenizer=self.processor,
            data_collator=self.collate_fn,
            compute_metrics=self.eval_compute_metrics_fn,
         )

        trainer.train()
        
        trainer.save_model("models/rt-detr/best_model")

        eval_logs = [entry for entry in trainer.state.log_history if "eval_loss" in entry or "eval_map" in entry]
        if eval_logs:
            pd.DataFrame(eval_logs).to_csv("./models/rt-detr/detr_metrics.csv", index=False)

        dummy_input = torch.randn(1, 3, self.img_size, self.img_size).to(self.device)
        torch.onnx.export(
            self.model,
            dummy_input,
            "models/rt-detr_model.onnx",
            export_params=True,
            opset_version=17,
            input_names=["input"],
            output_names=["output"]
            )
