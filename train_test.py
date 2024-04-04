from config import DEVICE
import numpy as np
import optuna
import os
import torch
from tqdm import tqdm

import utils

class Trainer():
    def __init__(self,
                 trial = None,
                 trial_num = None,
                 train_data = None,
                 test_data = None,
                 epochs = None,
                 model = None, 
                 optimizer = None, 
                 criterion = None, 
                 scheduler = None,
                 results_path = None
                 ):
        self.trial = trial
        self.trial_num = trial_num
        self.train_data = train_data
        self.test_data = test_data
        self.epochs = epochs
        self.model = model
        self.optimizer = optimizer 
        self.criterion = criterion 
        self.scheduler = scheduler
        self.results_path = results_path
        self.lr_all = []

    def train_epoch(self):
        batch_losses = []
        batch_accuracy = []

        self.model.train()

        prog_bar = tqdm(self.train_data,
                        total = len(self.train_data),
                        position = 0,
                        leave = False,
                        unit = " Batches")
        
        for i, (inputs, targets) in enumerate(prog_bar):
            self.optimizer.zero_grad()

            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            # Collect learning rate and batch_losses
            self.lr_all.append(self.optimizer.param_groups[0]['lr'])
            batch_losses.append(loss.item())

            # Calculate and collect batch accuracy
            outputs = torch.tensor(outputs.detach().cpu())
            targets = torch.tensor(targets.detach().cpu())

            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == targets).sum().item() / targets.size(0)
            batch_accuracy.append(accuracy)

        # Calculate epoch loss and accuracy
        epoch_loss = np.mean(batch_losses)
        epoch_accuracy = np.mean(batch_accuracy)
        
        return epoch_loss, epoch_accuracy

    def test_epoch(self):
        batch_losses = []
        batch_accuracy = []

        self.model.eval()

        prog_bar = tqdm(self.test_data,
                    total = len(self.test_data),
                    position = 0,
                    leave = False,
                    unit = " Batches")
        
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(prog_bar):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Collect batch losses
                batch_losses.append(loss.item())

                # Calculate batch accuracy
                outputs = torch.tensor(outputs.detach().cpu())
                targets = torch.tensor(targets.detach().cpu())
                
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == targets).sum().item() / targets.size(0)
                batch_accuracy.append(accuracy)
                
        # Calculate epoch loss and accuracy
        epoch_loss = np.mean(batch_losses)
        epoch_accuracy = np.mean(batch_accuracy)
        
        return epoch_loss, epoch_accuracy
    
    def fit(self):
        self.train_loss_list = []
        self.test_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []
        
        best_acc = float('-inf')
        
        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_epoch()
            test_loss, test_acc = self.test_epoch()
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}: "
                      f"Train Loss: {round(train_loss, 4)}, "
                      f"Train Accuracy: {round(train_acc, 4)}, "
                      f"Test Loss: {round(test_loss, 4)}, "
                      f"Test Accuracy: {round(test_acc, 4)}")

            self.train_loss_list.append(train_loss)
            self.train_acc_list.append(train_acc)
            self.test_loss_list.append(test_loss)
            self.test_acc_list.append(test_acc)
            
            # Checking for best performance
            if test_acc > best_acc:
                best_acc = test_acc
                best_loss = test_loss
                best_epoch = epoch
                
                # Saving best model
                self.save_model(self.trial_num, self.model)

            self.trial.report(test_acc, epoch)
            
            if self.trial.should_prune():
                raise optuna.TrialPruned()
        
        self.trial.set_user_attr('Test_loss', best_loss)
        self.trial.set_user_attr('Epoch', best_epoch)
        
        self.save_metrics()        
        
        return best_acc
    
    def save_model(self,
                   trial_num: int,
                   model):
        ckpt = {'model': model}
        torch.save(ckpt, os.path.join(self.results_path, 
                                      f"Trial_{trial_num}_model.pt"))
        
    def save_metrics(self):
        train_accuracies = [acc * 100 for acc in self.train_acc_list]
        test_accuracies = [acc * 100 for acc in self.test_acc_list]

        utils.save_plot_lr(learning_rates = self.lr_all,
                           results_path = self.results_path
                           )
        utils.save_metrics(loss_list = self.train_loss_list,
                           accuracy_list = train_accuracies,
                           results_path = self.results_path
                           )
        utils.save_metrics(loss_list = self.test_loss_list,
                           accuracy_list = test_accuracies,
                           results_path = self.results_path,
                           train = False
                           )
        utils.plot_metrics(train_list = self.train_loss_list,
                           test_list = self.test_loss_list,
                           results_path = self.results_path
                           )
        utils.plot_metrics(train_list = train_accuracies,
                           test_list = test_accuracies,
                           results_path = self.results_path,
                           loss = False
                           )