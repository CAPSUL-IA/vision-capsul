import timm
import torch
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from src.models.interfaces.model import IModel
from src.utils.utils import RunParser

from datetime import datetime

class ClassificationTimmModel(IModel):
    """
    Initializes the classification model with the given configuration,
    number of classes, classification type,
    and label encoding.

    Args:
        cfg: Configuration object containing model
            parameters and settings.
        num_classes: The number of unique classes in the dataset.
        label_encoding: Mapping of label names to indices.
        checkpoint (str): path of a model checkpoint.
        is_inference (bool): Bool to indicate if the model is used for inference.
    """
    def __init__(self, cfg: RunParser | None,
                 label_encoding: dict,
                 checkpoint=None,
                 is_inference=False,
                 *args,
                 **kwargs):
        self.label_encoding = label_encoding
        self.num_classes = len(self.label_encoding.keys())
        self.model_name = cfg.model if not is_inference else checkpoint['model_name'] 
        self.pretrained = cfg.pretrained if not is_inference else checkpoint['pretrained'] 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(self.num_classes, checkpoint, is_inference)
        self.batch_size = int(cfg.batch) if not is_inference else int(checkpoint['batch_size']) 
        self.multilabel = cfg.multilabel if not is_inference else int(checkpoint['multilabel']) 
        if not is_inference:
            self.img_size = cfg.img_size
            self.criterion = self.load_criterion()
            self.learning_rate = float(cfg.lr)
            self.optimizer = self.load_optimizer(cfg.optim)
            self.epochs = int(cfg.epoch)
            self.scheduler = self.load_scheduler(cfg.scheduler)
            self.patience = int(cfg.patience) if cfg.patience is not None else None
            self.path = "models/classificator" if cfg.overwrite else f"models/classificator_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            super().__init__(model=self.model, 
                         optimizer=self.optimizer, 
                         criterion=self.criterion,
                         scheduler=self.scheduler)

    def load_model(self, num_classes: int, checkpoint, is_inference):
        """
        Function to load the pretrained model to finetune to
        the given problem
        Args:
            checkpoint: path of a model checkpoint.
            num_classes: Number of classes in the problem.
            is_inference: Bool to indicate if the model is
             used for inference.
        Returns:
            torch.nn.Module: The loaded and adjusted model.
        """
        model = timm.create_model(
                self.model_name,
                pretrained=self.pretrained,
                num_classes=num_classes
        )
        if checkpoint is not None:
            model.load_state_dict(checkpoint['model_state_dict'])
        if is_inference:
            model.eval()
        model = model.to(self.device)
        return model

    def load_optimizer(self, optim: str):
        """
        Function to load the optimizer to use
        in training.
        Args:
            optim: name of the selected optimizer
        Returns:
            torch.optim.Optimizer: An initialized PyTorch
             optimizer ready for model training.
        """
        if optim == "AdamW":
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-6,
                weight_decay=0.0
            )
        elif optim == "Adam":
            optimizer = Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-6,
                weight_decay=0.0
            )
        else:
            optimizer = SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=0.0,
                momentum=0.9
            )
        return optimizer
    
    def load_criterion(self):
        """
        Selects and returns the appropriate loss function
        based on the classification type.
        Returns:
            An instance of a PyTorch loss function suitable
            for the specified classification type.
        """
        if self.multilabel:
            return torch.nn.BCEWithLogitsLoss(reduction='sum')
        return torch.nn.CrossEntropyLoss(reduction='sum')

    def load_scheduler(self, type_sch: str):
        """
        Function to load the scheduler to use in training.
        Args:
            type_sch: Name of the scheduler
        Returns:
            torch.optim.lr_scheduler._LRScheduler: An initialized
             PyTorch learning rate scheduler.
        """
        if type_sch == "Cosine":
            scheduler = CosineAnnealingLR(self.optimizer,
                                          self.epochs,
                                          eta_min=1e-6)
        else:
            scheduler = ReduceLROnPlateau(self.optimizer,
                                          'min')
        return scheduler
    
    def save_path(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'class_encoding': self.label_encoding,
            'img_size': self.img_size,
            'model_name': self.model_name,
            'pretrained': self.pretrained,
            'batch_size': self.batch_size,
            'multilabel': self.multilabel
        }, path)


    def train(self, dataset):
        super().train(dataset=dataset)


