import pytest
import pandas as pd
from torchvision import transforms
import os

from src.utils.utils import RunParser
from config.config import IMAGES_PATH, IMAGES_LABELS_PATH
from src.models.classes.model_factory import ModelFactory
from src.process.classes.dataset_factory import DatasetFactory

from src.models.classes.classification_timm_model import ClassificationTimmModel
from src.models.classes.detection_yolo_model import DetectionYoloModel

def test_cfg():
    cfg = RunParser("config/run.cfg")
    img_size = int(cfg.img_size)
    img_size = (img_size, img_size)
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])

    dataset = DatasetFactory(cfg=cfg,
                             csv_file=IMAGES_LABELS_PATH,
                             img_dir=IMAGES_PATH,
                             transform=transform)

    model = ModelFactory(cfg=cfg,
                         num_classes=dataset.num_classes,
                         label_encoding=dataset.label_encoding)
    
    if cfg.type == 'object_detection':
        assert isinstance(model, DetectionYoloModel)
    else:
        assert isinstance(model, ClassificationTimmModel)
        
def test_data():
    cfg = RunParser("config/run.cfg")
    assert os.path.exists("data/images/")
    assert os.path.exists("data/labels/")
    
    if cfg.type == 'object_detection':
        images = os.listdir("data/images/")
        labels = os.listdir("data/labels/")
        labels = [label.replace("txt","jpg") for label in labels]
        
        diff = [ref for ref in images if ref not in labels]
             
        assert len(diff) == 0
    else:
        images = os.listdir("data/images/")
        labels = pd.read_csv("data/labels/labels.csv")
        labels = labels["IMAGE_NAME"].tolist()
        
        diff = [ref for ref in images if ref not in labels]
             
        assert len(diff) == 0
    
