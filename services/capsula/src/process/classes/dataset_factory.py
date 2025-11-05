from src.utils.utils import RunParser, base_model
from src.process.classes.img_classification_dataset import ClassificationDataset
from src.process.classes.img_detection_yolo_dataset import ObjectDetectionYoloDataset
from src.process.classes.img_detection_detr_dataset import DETRDataset,ObjectDetectionDETRDataset
from config.config import YOLO_MODEL, TIMM_MODEL, DETR_MODEL, DEFAULT_MODEL

dataset_dict = {
    "classification": {
        TIMM_MODEL: ClassificationDataset
    },
    "object_detection": {
        YOLO_MODEL: ObjectDetectionYoloDataset,
        DETR_MODEL: DETRDataset 
    }
}

class DatasetFactory:
    def __new__(cls, cfg: RunParser, *args, **kwargs):
        task_str = cfg.type.lower()
        task_type = dataset_dict.get(task_str, None)
        stratify = cfg.stratify
        if task_type is None:
            raise NotImplementedError ("Not supported CAPSULE task type")
        
        model_name = base_model(cfg.model)
        dataset = task_type.get(model_name, None)
        if dataset is None:
            raise NotImplementedError (f"Dataset {model_name} not suported for '{task_str}' task")
        return dataset(stratify=stratify, stratify_ratio=cfg.eval_partition, *args, **kwargs)

        
