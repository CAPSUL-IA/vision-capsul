import os
import timm
import torch
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

import onnx
from onnxconverter_common import float16
from onnxruntime.quantization import quantize_dynamic, QuantType, preprocess

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

from src.models.interfaces.model import IModel
from src.utils.utils import RunParser

from datetime import datetime

import warnings

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
        self.amp = cfg.amp
        self.quantize = cfg.quantize
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

    def get_metrics_eval(self, val_metrics, epoch, train_loss):
        if not self.multilabel:
            val_metric = val_metrics['precision']
            print(f"Precision: {val_metric}\n")
            return {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_accuracy": val_metric
            }, val_metric
        else:
            val_metric = val_metrics["f1"]
            print(f"F1 Score: {val_metric}\n")
            return {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"]
            }, val_metric


    def generate_confusion_matrix(self, all_outputs, all_labels):
        predicted = torch.max(all_outputs, 1)[1]
        labels = all_labels.argmax(dim=1)

        cm = confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy())

        class_names = [k for k, v in sorted(self.label_encoding.items(), key=lambda x: x[1])]

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap='Blues', values_format='d')
        plt.title('Confusion Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.path,'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def calculate_metrics_eval(self, all_outputs, all_labels):
        if not self.multilabel:
            _, predicted = torch.max(all_outputs, 1)
            labels = all_labels.argmax(dim=1)

            correct = (predicted == labels).sum().item()

            self.generate_confusion_matrix(all_outputs, all_labels)

            return {'precision': correct / len(labels)}
        else:
            predictions = (torch.sigmoid(all_outputs) > 0.5).float()

            tp = (predictions * all_labels).sum().item()
            fp = (predictions * (1 - all_labels)).sum().item()
            fn = ((1 - predictions) * all_labels).sum().item()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            return {'precision': precision, 'recall': recall, 'f1': f1}

    def show_results(self, metrics):
        print("Resultados finales:")
        if not self.multilabel:
            print(f"Precision: {metrics['precision']}")
        else:
            print(f"Precision:  {metrics['precision']}")
            print(f"Recall: {metrics['recall']}")
            print(f"F1 Score: {metrics['f1']}")

    def export_to_onnx(self, model):
        onnx_path = os.path.join(self.path,"model.onnx")
        dummy_input = torch.randn(1, 3, int(self.img_size), int(self.img_size))
        torch.onnx.export(
            model.cpu(),
            dummy_input.cpu(),
            onnx_path,
            export_params=True,
            opset_version=13,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )

        if self.quantize:
            self.quantize_model(onnx_path)

    def quantize_model(self, onnx_path):
        quantize_path = os.path.join(self.path, "model_quantize.onnx")
        if self.quantize == "int8":
            preprocess.quant_pre_process(
                input_model_path=onnx_path,
                output_model_path=quantize_path,
                skip_optimization=False, 
                skip_onnx_shape=False,
                skip_symbolic_shape=False,
                auto_merge=True,
            )

            quantize_dynamic(
                quantize_path,
                quantize_path,
                weight_type=QuantType.QInt8
            )

            print("Model quantized to INT8 saved.") 
            
        elif self.quantize == "float16":
            model = onnx.load(onnx_path)
            
            model_fp16 = float16.convert_float_to_float16(model)

            onnx.save(model_fp16, quantize_path)
            print("Model quantized to float16 saved.") 
            
        else:
            warnings.warn(
                "The model has not been quantized. Only int8 or float16 quantization is supported.",
                UserWarning
            )
