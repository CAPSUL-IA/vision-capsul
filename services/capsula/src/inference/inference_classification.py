#!/usr/bin/env python3
# coding: utf-8
# Copyright (C) Solver Intelligent Analytics -All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by SolverIA <info@iasolver.com>, febrero 2024
#
import torch
import onnxruntime as ort

from src.inference.inference_dataset import InferenceDataset

class Timm_Inference:
    def __init__(self, model: str, path_image:str, img_size:int, multilabel: bool = False, classes: list = range(80)):
        """
        Initialize an instance of the YOLOv8 class.

        Args:
            onnx_model (str): Path to the ONNX model.
            input_image (str): Path to the images.
            confidence_thres (float): Confidence threshold for filtering detections.
            iou_thres (float): IoU threshold for non-maximum suppression.
        """
        self.model = model
        self.path = path_image
        self.img_size = img_size
        self.multilabel = multilabel

        # Load the class names from the COCO dataset
        self.classes = classes
        self.dataset = InferenceDataset(image_path=self.path, img_size=(self.img_size,self.img_size))

    def process_multiclass_output(self,outputs, label_encoding):
        """
        Processes the output of the model for a multiclass
        classification task.
        Args:
            outputs (torch.Tensor): The raw output from the model
             for a batch of images.
            label_encoding (dict): A dictionary mapping numerical
             labels to their corresponding string representation.
        Returns:
            tuple: A tuple containing a list of labels for each image
             and a tensor of probabilities associated with the
             predicted label for each image.
        """
        probs = torch.nn.functional.softmax(outputs, dim=1)
        image_probs_list, preds = torch.max(probs, dim=1)
        image_labels = [label_encoding[pred.item()] for pred in preds]
        return [image_labels], [image_probs_list]

    def process_multilabel_output(self, outputs, label_encoding):
        """
        Processes the output of the model for a multiclass
        classification task.
        Args:
            outputs (torch.Tensor): The raw output from the model
             for a batch of images.
            label_encoding (dict): A dictionary mapping numerical
             labels to their corresponding string representation.
        Returns:
            tuple: A tuple containing a list of labels for each image
             and a tensor of probabilities associated with the
             predicted label for each image.
        """
        image_labels, image_probs_list = [], []
        probs = torch.nn.functional.sigmoid(outputs)
        for probab in probs:
            index_preds = [i for i,prob in enumerate(probab) if prob>0.5]
            image_labels.append([label_encoding[i] for i in index_preds])
            image_probs_list.append([probab[i] for i in index_preds])

        return image_labels, image_probs_list

    def postprocess(self, outputs):
        final_preds, final_probs = [], []
        for out in outputs:
            out = torch.tensor(out[0], dtype=torch.float32)
            if self.multilabel:
                labels, probs = self.process_multilabel_output(out,
                                                          self.classes)
            else:
                labels, probs = self.process_multiclass_output(out,
                                                      self.classes)
            final_preds.extend(labels)
            final_probs.extend(probs)
        return final_preds, final_probs

    def show_results(self, infs, probs):    
        for i in range(len(self.dataset)):
            for j in range(len(infs[i])):
                cls = infs[i][j]
                prob = round(probs[i][j].item(), 4)
                img_name = self.dataset.imgs[i]
                print(f"Para la imagen {img_name} el modelo ha predicho la clase {cls} con una confianza de {prob}.")


    def main(self):

        provider = ["CUDAExecutionProvider","CPUExecutionProvider"]
        session = ort.InferenceSession(self.model, providers =provider)
        
        outputs = [session.run(None, {session.get_inputs()[0].name: self.dataset[i]}) for i in range(len(self.dataset))]       
        infs, probs = self.postprocess(outputs)
       
        self.show_results(infs, probs)

def do_inference(model, dataloader, label_encoding):
    """
    Performs inference on the given dataloader with the model.
    Args:
        model (torch.nn.Module): Trained model for inference.
        dataloader (torch.utils.data.DataLoader): DataLoader
         for batching the inference.
        label_encoding (dict): Dictionary mapping numerical labels
         to their real name
    Returns:
        final_preds (list): List of predicted classes for all instances.
        final_probs (list): List of probabilities associated with the
         predicted class for all instances.
    """
    final_preds = []
    final_probs = []
    for images in dataloader:
        images = images.to(model.device)
        with torch.no_grad():
            outputs = model.model(images)
            if model.multilabel:
                labels, probs = process_multilabel_output(outputs,
                                                      label_encoding)
            else:
                labels, probs = process_multiclass_output(outputs,
                                                      label_encoding)
            final_preds.extend(labels)
            final_probs.extend(probs)
    return final_preds, final_probs

