"""Configuration file of project variables that we want to have available everywhere and considered configuration."""

DATA_BASE_PATH = "data/"
DATA_FINAL_PATH = "data/final_data"
IMAGES_LABELS_PATH = "data/labels"
IMAGES_PATH = "data/images"
LOGS_PATH = "logs"
MODELS_SAVE_PATH = "models/last_model"

OBJ_DETECTION_MODEL_DIR = "models/object_detector"
OBJ_DETECTION_YAML_PATH = "models/dataset.yaml"

# COLOR DETECTOR
THR_RB = 16
THR_RG = 10
THR_BG = 16
THR_PIXELS = 0.01


# MODELS
YOLO_MODEL = "yolo"
DETR_MODEL = "detr"
TIMM_MODEL = "timm"
DEFAULT_MODEL = TIMM_MODEL

# YOLO ONNX PARAMS
CONF_THRESHOLD = 0.3
IOU_THRESHOLD=0.5
