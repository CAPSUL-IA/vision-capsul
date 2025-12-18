import os
import cv2
import torch

import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List, Tuple, Dict

class YOLOv8_Inference:
    def __init__(self, model: str, path_image:str, shape:int, conf: float, iou: float,
            classes: list = range(80), batch_size: int = 64):
        """
        Initialize the YOLOv8 inference class.

        Args:
            model (str): Path to the ONNX model.
            path_image (str): Directory containing input images.
            shape (int): Input size for the model.
            conf (float): Confidence threshold for filtering detections.
            iou (float): IoU threshold for non-maximum suppression (NMS).
            classes (list): List of class names or indices.
            draw(bool): draw the detections.
            batch_size(int): Number of images to process per batch.
        """

        self.model = model
        self.path = path_image

        if self.path:
            self.image_files = sorted([file for file in os.listdir(self.path) if
                     file.lower().endswith(('.png', '.jpg', '.jpeg'))]) 
        self.conf = conf
        self.iou = iou
        self.classes = classes
        self.batch_size = batch_size

        self.color_palette = self.get_colors()
        self.input_width, self.input_height = shape, shape
        
        providers = ["CUDAExecutionProvider","CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(self.model, providers = providers)

    def get_shape(self, batch_images: List[np.ndarray]):
        """
        Retrieve the original dimensions of a batch of images.

        Returns:
            tuple:
                - list[int]: Heights of input images.
                - list[int]: Widths of input images.
        """
        height, width=[],[]
        for img in batch_images:
            h,w=img.shape[:2]
            height.append(h)
            width.append(w)
            
        return height, width

    def get_colors(self):
        """
        Generate a distinct RGB color for each class.

        Returns:
            list[tuple[int, int, int]]: List of RGB color tuples.
        """
        cmap = plt.get_cmap('tab20')
        colors = [cmap(i % cmap.N) for i in range(len(self.classes))]
        colors = [(int(r*255),int(g*255),int(b*255)) for r,g,b,_ in colors]
        return colors
        
    def add_padding(self, img: np.ndarray, new_shape: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Apply letterboxing to maintain aspect ratio while resizing the image.

        Args:
            img (np.ndarray): Input image.
            new_shape (tuple[int, int]): Target shape after padding.

        Returns:
            tuple:
                - np.ndarray: Resized and padded image.
                - tuple[int, int]: Amount of padding (top, left) added.
        """
        shape = img.shape[:2]
        ratio_aspect = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        new_unpad = int(round(shape[1] * ratio_aspect)), int(round(shape[0] * ratio_aspect))
        pad_to_addw, pad_to_addh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding

        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(pad_to_addh - 0.1)), int(round(pad_to_addh + 0.1))
        left, right = int(round(pad_to_addw - 0.1)), int(round(pad_to_addw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        return img, (top, left)

    def draw_detections(self, img: np.ndarray, box: List[float], score: float, class_id: int) -> None:
        """
        Draw a bounding box and label on the image.

        Args:
            img (np.ndarray): Image where the detection will be drawn.
            box (list[float]): Bounding box [x, y, w, h].
            score (float): Confidence score.
            class_id (int): Predicted class ID.

        Returns:
            np.ndarray: Image with the detection drawn.
        """
        x0, y0, w, h = map(int, box) 
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), self.color_palette[class_id], 2)
        
        label = f"{self.classes[class_id]}: {score:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_x = x0
        label_y = y0 - 10 if y0 - 10 > label_height else y0 + 10
        cv2.rectangle(
            img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), self.color_palette[class_id], cv2.FILLED
        )
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return img

    def preprocess(self, batch_images: List[np.ndarray]) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """
        Preprocess a batch of images for model inference.
        Includes: color conversion, letterboxing, normalization, and layout change.

        Returns:
            tuple:
                - list[np.ndarray]: Preprocessed images in model input format.
                - list[tuple[int, int]]: Padding applied to each image.
        """
        image_data, pad = [],[]
        for img in batch_images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img, p = self.add_padding(img, (self.input_width, self.input_height))

            img_data = np.array(img) / 255.0
            img_data = np.transpose(img_data, (2, 0, 1))
            img_data = np.expand_dims(img_data, axis=0).astype(np.float32)
            
            image_data.append(img_data)
            pad.append(p)
        return image_data, pad

    def nms_numpy(self, boxes, scores, iou_threshold):
        """
        Non-maximum suppression NumPy.
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def detect_and_nms(self, out_i, pad_item, img_shape):
        h, w = img_shape
        scale = min(self.input_height / h, self.input_width / w)
        pad_top, pad_left = pad_item

        boxes_raw = out_i[:, :4]
        class_scores = out_i[:, 4:]

        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(class_scores.shape[0]), class_ids]

        mask = scores >= self.conf
        if not mask.any():
            return []

        boxes_raw = boxes_raw[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        # Coordenadas originales
        cx = (boxes_raw[:, 0] - pad_left) / scale
        cy = (boxes_raw[:, 1] - pad_top) / scale
        bw = boxes_raw[:, 2] / scale
        bh = boxes_raw[:, 3] / scale

        left = cx - bw / 2
        top = cy - bh / 2
        right = left + bw
        bottom = top + bh

        boxes_xyxy = np.stack([left, top, right, bottom], axis=1)

        # NMS en NumPy
        keep = self.nms_numpy(boxes_xyxy, scores, self.iou)
        boxes_final = boxes_xyxy[keep]
        scores_final = scores[keep]
        class_ids_final = class_ids[keep]

        # Convertir a formato (left, top, w, h)
        boxes_out = np.stack([
            boxes_final[:, 0],
            boxes_final[:, 1],
            boxes_final[:, 2] - boxes_final[:, 0],
            boxes_final[:, 3] - boxes_final[:, 1]
        ], axis=1)

        return [(boxes_out[i], float(scores_final[i]), int(class_ids_final[i])) for i in range(len(keep))]

    def postprocess(self, output: List[List[np.ndarray]], pad: List[Tuple[int, int]], batch_images: List[np.ndarray]) -> Dict[str, List[Dict]]:
        """
        Convert raw model output into final bounding boxes using NMS.
        Args:
            output (list): Raw outputs from the ONNX model.
            pad (list[tuple]): Padding added during preprocessing.
            batch_images(list[np.ndarray]): Batch of original images.
        Returns:
            - None
        """
        batch_output = np.transpose(output[0], (0, 2, 1))  # B x N x 85
        for i, img in enumerate(batch_images):
            detections = self.detect_and_nms(batch_output[i], pad[i], img.shape[:2])

            for box, score, class_id in detections:
                img = self.draw_detections(img, box.tolist(), score, class_id)
            batch_images[i] = img

    def main(self):
        """
        Run the complete inference pipeline in batches:
        preprocess, ONNX inference and postprocess.

        Returns:
           - None 
        """

        for i in range(0, len(self.image_files), self.batch_size):
            batch_files = self.image_files[i:i+self.batch_size]
            self.img = [cv2.imread(os.path.join(self.path,f)) for f in batch_files]

            img_data, pad = self.preprocess(self.img)

            batch_input = np.concatenate(img_data, axis=0)
            outputs = self.session.run(None, {self.session.get_inputs()[0].name: batch_input})

            self.postprocess(outputs, pad, self.img)

            os.makedirs(f"{self.path}/result/",exist_ok=True)
            for j, f in enumerate(batch_files):
                cv2.imwrite(os.path.join(f"{self.path}/result/", f), self.img[j])
