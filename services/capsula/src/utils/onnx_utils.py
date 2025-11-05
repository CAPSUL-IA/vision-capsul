#!/usr/bin/env python3
# coding: utf-8
# Copyright (C) Solver Intelligent Analytics -All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by SolverIA <info@iasolver.com>, marzo 2024
#
import numpy as np
import cv2
from PIL import Image


def nms(boxes, scores, iou_threshold):
    """
    Performs non-maximum suppression (NMS) on the boxes according to their
    intersection-over-union (IoU).
    Args:
        boxes (np.ndarray): Array of bounding boxes (x1, y1, x2, y2).
        scores (np.ndarray): Array of scores for each box.
        iou_threshold (float): Threshold for IoU to decide whether boxes overlap.
    Returns:
        list: Indices of boxes kept after applying NMS.
    """
    sorted_indices = np.argsort(scores)[::-1]
    keep_boxes = []
    while sorted_indices.size > 0:
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])
        keep_indices = np.where(ious < iou_threshold)[0]
        sorted_indices = sorted_indices[keep_indices + 1]
    return keep_boxes


def multiclass_nms(boxes, scores, class_ids, iou_threshold):
    """
    Applies Non-Maximum Suppression (NMS) separately for each class.
    Args:
        boxes (np.ndarray): Array of bounding boxes.
        scores (np.ndarray): Array of scores for each box.
        class_ids (np.ndarray): Array of class IDs for each box.
        iou_threshold (float): IOU threshold for NMS.
    Returns:
        list: Indices of boxes kept after applying NMS.
    """
    unique_class_ids = np.unique(class_ids)
    keep_boxes = []
    for class_id in unique_class_ids:
        class_indices = np.where(class_ids == class_id)[0]
        class_boxes = boxes[class_indices, :]
        class_scores = scores[class_indices]
        class_keep_boxes = nms(class_boxes, class_scores, iou_threshold)
        keep_boxes.extend(class_indices[class_keep_boxes])
    return keep_boxes


def compute_iou(box, boxes):
    """
    Computes Intersection Over Union (IoU) between a box and a list of boxes.
    Args:
        box (numpy.ndarray): Single bounding box.
        boxes (numpy.ndarray): Array of bounding boxes to compare against.
    Returns:
        numpy.ndarray: IoU values between `box` and each box in `boxes`.
    """
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area
    iou = intersection_area / union_area
    return iou


def xywh2xyxy(box):
    """
    Converts bounding boxes from (x_center, y_center, width, height) to
    (x_min, y_min, x_max, y_max) format.
    Args:
        box (numpy.ndarray): Array containing bounding boxes in (x, y, w, h) format.
    Returns:
        numpy.ndarray: Converted bounding boxes in (x_min, y_min, x_max, y_max) format.
    """
    processed_box = np.copy(box)
    processed_box[..., 0] = box[..., 0] - box[..., 2] / 2
    processed_box[..., 1] = box[..., 1] - box[..., 3] / 2
    processed_box[..., 2] = box[..., 0] + box[..., 2] / 2
    processed_box[..., 3] = box[..., 1] + box[..., 3] / 2
    return processed_box


def xyxy2xywh(box):
    """
    Converts bounding boxes from (x_min, y_min, x_max, y_max) to
    (x_center, y_center, width, height) format.
    Args:
        box (numpy.ndarray): Array containing bounding boxes in (x_min, y_min, x_max, y_max) format.
    Returns:
        numpy.ndarray: Converted bounding boxes in (x_center, y_center, width, height) format.
    """
    processed_box = np.copy(box)
    processed_box[..., 0] = (box[..., 0] + box[..., 2]) / 2
    processed_box[..., 1] = (box[..., 1] + box[..., 3]) / 2
    processed_box[..., 2] = box[..., 2] - box[..., 0]
    processed_box[..., 3] = box[..., 3] - box[..., 1]
    return processed_box


def detections_to_draw(image_paths, detections_dic, id_label, label_id):
    """
    Processes detected objects in images and displays the images with
    detections drawn.
    This function reads images specified in `image_paths`, uses the detection
    results provided in `detections_dic`, and visualizes each detection by
    drawing bounding boxes, classes, and confidence scores on the images.
    Args:
        image_paths (list): List of paths to the images.
        detections_dic (dict): Dictionary containing detection results with
            keys 'detections', where each detection includes 'labels' with
            properties 'x', 'y', 'width', 'height', 'confidence', and 'class'.
        id_label (dict): Mapping of class IDs to class labels.
        label_id (dict): Mapping of class labels to class IDs.
    The function does not return any value but displays each image with its
    detections in a window.
    """
    for img_index, det in enumerate(detections_dic['detections']):
        image = cv2.imread(image_paths[img_index])
        boxes = []
        scores = []
        class_ids = []
        for label in det['labels']:
            x_center, y_center, width, height = (label['x'],
                                                 label['y'],
                                                 label['width'],
                                                 label['height'])
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            boxes.append([x1, y1, x2, y2])
            scores.append(label['confidence'])
            class_ids.append(label_id[label['class']])
        boxes = np.array(boxes)
        scores = np.array(scores)
        class_ids = np.array(class_ids)
        drawn_image = draw_detections(image, boxes, scores,
                                      class_ids, id_label)
        cv2.imshow(f'Detections {img_index}', drawn_image)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_detections(image, boxes, scores,
                    class_ids, id_label,
                    mask_alpha=0.4):
    """
    Draws detections (bounding boxes and labels) on an image.
    Args:
        image (numpy.ndarray): Image on which to draw the detections.
        boxes (numpy.ndarray): Bounding boxes for detections.
        scores (numpy.ndarray): Scores for each detection.
        class_ids (numpy.ndarray): Class IDs for each detection.
        id_label (dict): Mapping from class IDs to labels.
        mask_alpha (float, optional): Alpha value for masks.
    Returns:
        numpy.ndarray: Image with detections drawn.
    """
    rng = np.random.default_rng(3)
    colors = rng.uniform(0, 255, size=(len(id_label.keys()), 3))
    det_img = image.copy()
    img_height, img_width = image.shape[:2]
    boxes *= np.array([img_width, img_height, img_width, img_height])
    font_size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)
    det_img = draw_masks(det_img, boxes, class_ids, colors, mask_alpha)
    # Draw bounding boxes and labels of detections
    for class_id, box, score in zip(class_ids, boxes, scores):
        color = colors[class_id]
        draw_box(det_img, box, color)
        label = id_label[class_id]
        caption = f'{label} {int(score * 100)}%'
        draw_text(det_img, caption, box, color, font_size, text_thickness)
    return det_img


def draw_cam_detection(detections_dic, image, label_id, id_label,
                       mask_alpha=0.4):
    """
    Draws detections (bounding boxes and labels) on an image.
    Args:
        image (numpy.ndarray): Image on which to draw the detections.
        id_label (dict): Mapping from class IDs to labels.
        mask_alpha (float, optional): Alpha value for masks.
    Returns:
        numpy.ndarray: Image with detections drawn.
    """
    detection = detections_dic['detections']
    if detection:
        det = detection[0]
        boxes = []
        scores = []
        class_ids = []
        for label in det['labels']:
            x_center, y_center, width, height = (label['x'],
                                                 label['y'],
                                                 label['width'],
                                                 label['height'])
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            boxes.append([x1, y1, x2, y2])
            scores.append(label['confidence'])
            class_ids.append(label_id[label['class']])
        boxes = np.array(boxes)
        scores = np.array(scores)
        class_ids = np.array(class_ids)
        rng = np.random.default_rng(3)
        colors = rng.uniform(0, 255, size=(len(id_label.keys()), 3))
        det_img = image.copy()
        img_height, img_width = image.shape[:2]
        boxes *= np.array([img_width, img_height, img_width, img_height])
        font_size = min([img_height, img_width]) * 0.0006
        text_thickness = int(min([img_height, img_width]) * 0.001)
        det_img = draw_masks(det_img, boxes, class_ids, colors, mask_alpha)
        # Draw bounding boxes and labels of detections
        for class_id, box, score in zip(class_ids, boxes, scores):
            color = colors[class_id]
            draw_box(det_img, box, color)
            label = id_label[class_id]
            caption = f'{label} {int(score * 100)}%'
            draw_text(det_img, caption, box, color, font_size, text_thickness)
        return det_img
    return image


def draw_box(image, box,
             color=(0, 0, 255),
             thickness=2):
    """
    Draws a single bounding box on an image.
    Args:
        image (numpy.ndarray): Image on which to draw.
        box (numpy.ndarray): Coordinates of the box to draw (x1, y1, x2, y2).
        color (tuple, optional): Color of the box. Defaults to (0, 0, 255).
        thickness (int, optional): Thickness of the box. Defaults to 2.
    Returns:
        numpy.ndarray: Image with the box drawn.
    """
    x1, y1, x2, y2 = box.astype(int)
    return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def draw_text(image, text, box,
              color=(0, 0, 255),
              font_size=0.001, text_thickness=2):
    """
    Draws text on an image above a specified box.
    Args:
        image (numpy.ndarray): Image on which to draw text.
        text (str): Text to draw.
        box (numpy.ndarray): Box above which to draw text.
        color (tuple, optional): Color of the text background.
        font_size (float, optional): Font size of the text.
        text_thickness (int, optional): Thickness of the text.
    Returns:
        numpy.ndarray: Image with the text drawn.
    """
    x1, y1, _, _ = box.astype(int)
    (tw, th), _ = cv2.getTextSize(text=text,
                                  fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                  fontScale=font_size,
                                  thickness=text_thickness)
    th = int(th * 1.2)
    cv2.rectangle(image, (x1, y1),
                  (x1 + tw, y1 - th), color, -1)

    return cv2.putText(image, text, (x1, y1),
                       cv2.FONT_HERSHEY_SIMPLEX, font_size,
                       (255, 255, 255), text_thickness,
                       cv2.LINE_AA)


def draw_masks(image, boxes, classes, colors,
               mask_alpha=0.3):
    """
    Draws semi-transparent masks over detected objects in an image.

    Args:
        image (numpy.ndarray): Original image as a numpy array.
        boxes (numpy.ndarray): Array of bounding boxes for each detected object,
            with each box defined by its top-left and bottom-right coordinates.
        classes (numpy.ndarray): Array of class IDs for each detected object.
        colors: List of colors for each class ID.
        mask_alpha (float, optional): Transparency level of the masks.
            Defaults to 0.3.
    Returns:
        numpy.ndarray: Image with semi-transparent masks drawn over detected
            objects.
    """
    mask_img = image.copy()
    for box, class_id in zip(boxes, classes):
        color = colors[class_id]
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
    return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)


def adapt_format(images):
    """
    Converts a list of image paths or PIL.Image objects to numpy arrays.
    Args:
        images (list): List of image paths (str),  PIL.Image objects
         or np.arrays.
    Returns:
        list: List of images converted to numpy arrays.
    """
    prepared_images = []
    for img in images:
        if isinstance(img, str):
            img = cv2.imread(img)
        elif isinstance(img, Image.Image):
            img = np.array(img)
        prepared_images.append(img)
    return prepared_images
