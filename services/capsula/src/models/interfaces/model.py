import torch
import os
import pandas as pd
from abc import ABCMeta, abstractmethod
from tqdm import tqdm
from torch.utils.data import DataLoader

class IModel(metaclass=ABCMeta):
    def __init__(self, model, optimizer=None, criterion=None, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler=scheduler

    @abstractmethod
    def load_model(self):
        pass

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
               
                with torch.amp.autocast(device_type = "cuda", enabled = self.amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                total += targets.size(0)
                
                if not is_eval:
                    loss.backward()    
                    optimizer.step()
                    optimizer.zero_grad()
                    epoch_loss += loss.item()

                if is_eval:
                    all_outputs.append(outputs.detach())
                    all_labels.append(targets.detach()) 

        if is_eval:
            all_outputs = torch.cat(all_outputs)
            all_labels = torch.cat(all_labels)
            
            return self.calculate_metrics_eval(all_outputs, all_labels)

        return epoch_loss / total
               
    
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
            
            metrics_validation, val_metric = self.get_metrics_eval(val_metrics, epoch, train_loss)
            metrics.append(metrics_validation)

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
        self.show_results(final_val_metrics)

        self.export_to_onnx(self.model)
