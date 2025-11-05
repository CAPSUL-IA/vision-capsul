from src.utils.utils import RunParser, base_model
from src.models.classes.classification_timm_model import ClassificationTimmModel
from src.models.classes.detection_yolo_model import DetectionYoloModel
from src.models.classes.detection_detr_model import DetectionDetrModel

from config.config import YOLO_MODEL, DETR_MODEL, TIMM_MODEL, DEFAULT_MODEL


models_dict = {
    "classification": {
        TIMM_MODEL: ClassificationTimmModel,
        YOLO_MODEL: "TODO"
    },
    "object_detection": {
        YOLO_MODEL: DetectionYoloModel,
        DETR_MODEL: DetectionDetrModel
    }
}

class ModelFactory:
    def __new__(cls, cfg: RunParser, *args, **kwargs):
        task_str = cfg.type.lower()
        task_type = models_dict.get(task_str, None)
        if task_type is None:
            raise NotImplementedError ("Not supported CAPSULE task type")
        
        model_name = base_model(cfg.model)
        model = task_type.get(model_name, None)
        if model is None:
            raise NotImplementedError (f"Model {model_name} not suported for '{task_str}' task")
        return model(cfg, *args, **kwargs)

    
