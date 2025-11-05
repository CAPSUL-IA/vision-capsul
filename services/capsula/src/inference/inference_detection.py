from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import onnxruntime as ort
import os
import numpy as np
import time

class YOLOv8_Inference:
    def __init__(self, model: str, path_image:str, image: List[str], conf: float, iou: float, shape:int, classes: list = range(80)):
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
        self.image = image
        self.conf = conf
        self.iou = iou

        # Load the class names from the COCO dataset
        self.classes = classes

        # Generate a color palette for the classes
        self.color_palette = self.get_colors()
        self.input_width, self.input_height = shape, shape
        
        self.img = [cv2.imread(os.path.join(self.path,img)) for img in self.image]
        self.img_height, self.img_width = self.get_shape()

    def get_shape(self):
        height, width=[],[]
        for img in self.img:
            h,w=img.shape[:2]
            height.append(h)
            width.append(w)
            
        return height, width

    def get_colors(self):
        cmap = plt.get_cmap('tab20')
        colors = [cmap(i % cmap.N) for i in range(len(self.classes))]
        
        colors = [(int(r*255),int(g*255),int(b*255)) for r,g,b,_ in colors]
        return colors
        
    def add_padding(self, img: np.ndarray, new_shape: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Computes the image so it has the aspect yolo uses for training.
        """
        shape = img.shape[:2]
        ratio_aspect = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * ratio_aspect)), int(round(shape[0] * ratio_aspect))
        pad_to_addw, pad_to_addh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding

        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(pad_to_addh - 0.1)), int(round(pad_to_addh + 0.1))
        left, right = int(round(pad_to_addw - 0.1)), int(round(pad_to_addw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        return img, (top, left)

    def draw_detections(self, img: np.ndarray, box: List[float], score: float, class_id: int) -> None:
        """
        Return the image with the bounding boxes.
        """
        # Coordinates
        x0, y0, w, h = box
        cv2.rectangle(img, (int(x0), int(y0)), (int(x0 + w), int(y0 + h)), self.color_palette[class_id], 2)
        
        #Draw label with its confidence
        label = f"{self.classes[class_id]}: {score:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_x = x0
        label_y = y0 - 10 if y0 - 10 > label_height else y0 + 10
        cv2.rectangle(
            img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), self.color_palette[class_id], cv2.FILLED
        )
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return img
        
    def preprocess(self) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """
        Tranform the image for making inference
        """
        image_data, pad = [],[]
        for i in range(len(self.image)):
            img = cv2.cvtColor(self.img[i], cv2.COLOR_BGR2RGB)

            #Padding
            img, p = self.add_padding(img, (self.input_width, self.input_height))

            # Normalize, transpose and expand
            img_data = np.array(img) / 255.0
            img_data = np.transpose(img_data, (2, 0, 1))
            img_data = np.expand_dims(img_data, axis=0).astype(np.float32)
            
            image_data.append(img_data)
            pad.append(p)
        return image_data, pad

    def postprocess(self, output: List[List[np.ndarray]], pad: List[Tuple[int, int]]) -> List[np.ndarray]:
        """
        Perform post-processing on the model's output to extract and visualize detections.

        This method processes the raw model output to extract bounding boxes, scores, and class IDs.
        It applies non-maximum suppression to filter overlapping detections and draws the results on the input image.

        Args:
            input_image (np.ndarray): The input image.
            output (List[np.ndarray]): The output arrays from the model.
            pad (Tuple[int, int]): Padding values (top, left) used during letterboxing.

        Returns:
            (np.ndarray): The input image with detections drawn on it.
        """
        for out in range(len(output)):
            # Transpose and squeeze the output to match the expected shape
            outputs = np.transpose(np.squeeze(output[out][0]))

            # Get the number of rows in the outputs array
            rows = outputs.shape[0]

            # Lists to store the bounding boxes, scores, and class IDs of the detections
            boxes = []
            scores = []
            
            # Calculate the scaling factors for the bounding box coordinates
            scale = min(self.input_height / self.img_height[out], self.input_width / self.img_width[out])
            outputs[:, 0] -= pad[out][1]
            outputs[:, 1] -= pad[out][0]

            for i in range(rows):
                #Score must be higher than threshold confidence
                score = outputs[i][4:][0]
                if score >= self.conf:
                    x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                    left = int((x - w / 2) / scale)
                    top = int((y - h / 2) / scale)
                    width = int(w / scale)
                    height = int(h / scale)

                    scores.append(score)
                    boxes.append([left, top, width, height])
            correct = cv2.dnn.NMSBoxes(boxes, scores, self.conf, self.iou)
            #Draw the detections
            for i in correct:
                box = boxes[i]
                score = scores[i]
                self.img[out] = self.draw_detections(self.img[out], box, score, 0)

    def main(self) -> np.ndarray:
        """
        Prepare the image, run inference and return the result
        """
        os.makedirs(f"{self.path}/result/",exist_ok=True)
        provider = ["CUDAExecutionProvider","CPUExecutionProvider"]
        session = ort.InferenceSession(self.model, providers =provider)
        img_data, pad = self.preprocess()
        
        outputs = [session.run(None, {session.get_inputs()[0].name: img_data[i]}) for i in range(len(img_data))]       
        self.postprocess(outputs, pad)
        
        for i in range(len(self.img)):
            cv2.imwrite(os.path.join(f"{self.path}/result/",self.image[i]),self.img[i])
