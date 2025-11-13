import torch
import os
import pandas as pd
from abc import ABCMeta, abstractmethod
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class IModel(metaclass=ABCMeta):
    def __init__(self, model, optimizer=None, criterion=None, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler=scheduler

    @abstractmethod
    def load_model(self):
        pass

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

    def run_batch(self, dataloader, is_eval: bool):
        """
        Runs a single batch of data through the model,
        performing either training or evaluation.

        Args:
            dataloader (DataLoader): The DataLoader providing batches of
            data for training or evaluation.
            is_eval: Flag indicating whether the model is
            being evaluated.

        Returns:
            tuple: A tuple containing concatenated outputs of the model,
            concatenated labels, and the total loss for the epoch.
        """
        model = self.model
        optimizer = self.optimizer
        criterion = self.criterion
        if is_eval:
            model.eval()
            description = "Evaluating"
        else:
            model.train()
            description = "Training set"
        epoch_loss = 0.0
        all_outputs = []
        all_labels = []
        correct, total = 0, 0
        # in evaluation no need of updating gradients
        with torch.set_grad_enabled(not is_eval):
            for batch in tqdm(dataloader, desc=description, unit="batch"):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total += targets.size(0)
                if not is_eval:
                    loss.backward()
                    
                    optimizer.step()
                    
                    optimizer.zero_grad()
                    
                    epoch_loss += loss.item()
                if is_eval and not self.multilabel:
                    _, predicted = torch.max(outputs.data, 1)
                    labels = targets.argmax(dim=1)
                    correct += (predicted == labels).sum().item()
                
                all_outputs.append(outputs.detach())
                all_labels.append(targets.detach())
                
        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)

        if not self.multilabel or not is_eval:
            #Métricas de evaluación
            if is_eval:
                self.generate_confusion_matrix(all_outputs, all_labels)
                return correct / total 
            #Perdida de entrenamiento
            return epoch_loss / total

        predictions = (torch.sigmoid(all_outputs) > 0.5).float()
        
        tp = (predictions * all_labels).sum().item()
        fp = (predictions * (1 - all_labels)).sum().item()
        fn = ((1 - predictions) * all_labels).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        return metrics
        
    
    @abstractmethod
    def train(self, dataset):
        """
        Trains and evaluates the model using the provided datasets. It
        saves best model and last model. Also generates graphics and logs
        files regarding model perfomance during train/eval phases.
        Args:
            dataset (Dataset): The dataset for training the model.
        """
        os.makedirs(self.path, exist_ok=True)
        best_model_path = os.path.join(self.path,"best_model") 
        last_model_path = os.path.join(self.path,"last_model")
        epochs = self.epochs
        batch = self.batch_size
        train_dataloader = DataLoader(dataset.train,
                                    batch_size=batch)
        val_dataloader = DataLoader(dataset.val,
                                    batch_size=batch)

        best_val_metric, best_epoch, not_improve = 0.0, 0, 0
        metrics = []
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1} of {epochs}")
            train_loss = self.run_batch(dataloader=train_dataloader,is_eval=False)
            
            print(f"Perdida: {train_loss}\n")
            val_metrics = self.run_batch(dataloader=val_dataloader,is_eval=True)
            
            if not self.multilabel:
                val_metric = val_metrics
                print(f"Precision: {val_metric}\n")
                metrics.append({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_accuracy": val_metric
                })
            else:
                val_metric = val_metrics["f1"]
                print(f"F1 Score: {val_metric}\n")
                metrics.append({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "val_f1": val_metrics["f1"]
                })
            if val_metric > best_val_metric:
                not_improve = 0
                best_val_metric = val_metric
                best_epoch = epoch

                self.save_path(best_model_path)
                
            else:
                not_improve += 1
                if self.patience and not_improve >= self.patience:
                    break
            
            self.scheduler.step()
       
        self.save_path(last_model_path)

        df = pd.DataFrame(metrics)
        df.to_csv(os.path.join(self.path,"metrics.csv"), index=False)

        print("\nValidación del mejor modelo:")
        checkpoint = torch.load(best_model_path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        final_val_metrics = self.run_batch(dataloader=val_dataloader, is_eval=True)
        print("Resultados finales:")
        if not self.multilabel:
            print(f"Precision: {final_val_metrics}")
        else:
            print(f"Precision:  {final_val_metrics['precision']}")
            print(f"Recall: {final_val_metrics['recall']}")
            print(f"F1 Score: {final_val_metrics['f1']}")
