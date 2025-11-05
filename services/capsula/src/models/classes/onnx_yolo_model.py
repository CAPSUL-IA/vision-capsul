import cv2
import numpy as np
import onnxruntime

from config.config import CONF_THRESHOLD, IOU_THRESHOLD
from src.utils.onnx_utils import multiclass_nms, xywh2xyxy, adapt_format

class OnnxYoloModel:
    """
    A class for loading and running inferences using the YOLO model
    in ONNX format.

    Attributes:
        model_name (str): Type of model (Yolo or Detr)-
        conf_threshold (float): Threshold for filtering out predictions
            with low confidence.
        iou_threshold (float): Intersection Over Union threshold for
            non-maximum suppression.
        id_label (int): Label ID for detected objects.
        input_width (int): Width to resize input images.
        input_height (int): Height to resize input images.
        batch_size (int): Number of images to process in a batch.
        session (onnxruntime.InferenceSession): ONNX runtime session for
            model inference.
    """
    def __init__(self, path, label_encoding, input_size, batch_size,
                 model_name, conf_thres=CONF_THRESHOLD, iou_thres=IOU_THRESHOLD):
        """
        Initializes the YOLOv8 ONNX model with given parameters.

        Args:
            path (str): Path to the ONNX model file.
            label_encoding (dict): Identifier label for detected objects.
            input_size (int): Size to resize input images for the model.
            batch_size (int): Batch size for processing images.
            conf_thres (float): Confidence threshold for detections.
            iou_thres (float): IOU threshold for non-maximum suppression.
        """
        self.label_encoding = label_encoding
        self.input_width = input_size
        self.input_height = input_size
        self.batch_size = batch_size
        self.model_name = model_name
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

    def __call__(self, images):
        """
        Detect objects in images using the YOLOv8 ONNX model.
        Args:
            images (list): List of images in numpy array format.

        Returns:
            tuple: (all_boxes, all_scores, all_class_ids) where
                all_boxes is a list of bounding boxes,
                all_scores is a list of confidence scores, and
                all_class_ids is a list of class IDs for the detections.
        """
        np_images = adapt_format(images)
        batches = [np_images[i:i + self.batch_size] for i in range(0,
                                                                   len(np_images),
                                                                   self.batch_size)]
        all_boxes = []
        all_scores = []
        all_class_ids = []
        for batch in batches:
            boxes, scores, class_ids = self.detect_objects(batch)
            all_boxes.extend(boxes)
            all_scores.extend(scores)
            all_class_ids.extend(class_ids)

    def initialize_model(self, path):
        """
        Initialize the ONNX runtime session for the model.
        Args:
            path (str): Path to the ONNX model file.
        """
        self.session = onnxruntime.InferenceSession(path,
                                                    providers=onnxruntime.get_available_providers())
        # Set model info
        self.set_input_details()
        self.set_output_details()

    def set_input_details(self):
        """
        Retrieve and store the model's input details.
        """
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = model_inputs[2].shape
        self.input_width = model_inputs[3].shape

    def set_output_details(self):
        """
        Retrieve and store the model's output details.
        """
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def detect_objects(self, images):
        """
        Detect objects in a batch of images.
        Args:
            images (list): Batch of images as numpy arrays.
        Returns:
            tuple: (boxes, scores, class_ids) for detected objects.
        """
        input_tensor = self.img2tensor(images)
        outputs = self.inference(input_tensor)
        self.boxes, self.scores, self.class_ids = self.proces_output(outputs)
        return self.boxes, self.scores, self.class_ids
    
    def img2tensor(self, images):
        """
        Preprocess images for model input.
        Args:
            images (list): List of images as numpy arrays.

        Returns:
            numpy.ndarray: Preprocessed images as a numpy array.
        """
        input_batch = []
        for image in images:
            input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_img = cv2.resize(input_img, (self.input_width, self.input_height))
            input_img = input_img / 255.0
            input_img = input_img.traspose(2, 0, 1)
            input_batch.append(input_img)
        input_tensor = np.stack(input_batch, axis=0).astype(np.float32)
        return input_tensor
    
    def inference(self, input_tensor):
        """
        Perform inference on the input tensor.
        Args:
            input_tensor (numpy.ndarray): Input tensor for the model.
        Returns:
            list: list of numpy.ndarray model outputs.
        """
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        return outputs
    
    def process_output(self, output):
        """
        Process the output from the model inference.
        Args:
            output (list): list of numpy.ndarray model outputs.
        Returns:
            tuple: Processed (boxes, scores, class_ids) for detected objects.
        """
        final_boxes = []
        final_scores = []
        final_class_ids = []
        predictions_batch = np.squeeze(output[0].T)
        if len(predictions_batch.shape) == 2:
            prediction_batch = np.expand_dims(predictions_batch, axis=-1)
        for batch_idx in range(prediction_batch.shape[-1]):
            # Filter out object confidence scores below threshold
            predictions = prediction_batch[:, :, batch_idx]
            if "detr" in self.model_name:
                predictions = predictions.T
            # Extract the max class score from the current image
            scores = np.max(predictions[:, 4:], axis=1)
            predictions = predictions[scores > self.conf_threshold, :]
            if len(scores) != 0:
                # Get the class with the highest confidence
                class_ids = np.argmax(predictions[:, 4:], axis=1)
                # Get the bounding boxes for each object
                boxes = self.extract_boxes(predictions)
                # Apply non-maxima suppresion to suppress weak, overlapping bounding boxes
                # nms does not support batched inference, loop to do postprocessing is needed
                indices = multiclass_nms(boxes, scores, class_ids, self.iou_threshold)
                final_boxes.append(boxes[indices])
                final_scores.append(scores[indices])
                final_class_ids.append(class_ids[indices])
        return final_boxes, final_scores, final_class_ids

    def extract_boxes(self, predictions):
        """
        Extract bounding boxes from model predictions.
        Args:
            predictions (numpy.ndarray): Predictions from the model.
        Returns:
            numpy.ndarray: Extracted bounding boxes.
        """
        # Extract the first 4 columns (x,y,w,h) from the predictions
        boxes = predictions[:, :4]
        if "detr" not in self.model_name:
            boxes = self.rescale_boxes(boxes)
        boxes = xywh2xyxy(boxes)
        return boxes
    
    def rescale_boxes(self, boxes):
        """
        Rescale bounding boxes to the original image size.
        Args:
            boxes (numpy.ndarray): Normalized bounding boxes.
        Returns:
            numpy.ndarray: Rescaled bounding boxes.
        """
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        return boxes
