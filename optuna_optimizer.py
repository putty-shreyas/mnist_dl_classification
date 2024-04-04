import joblib
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import config
from config import DEVICE
from hyperparameters import Hyperparameters
import model as Network
import train_test
import utils

class OptunaOptimizer():
    def __init__(self,
                 main_path,
                 data_path,
                 results_path,
                 subset,
                 saved_study,
                 n_trials
                 ):
        
        self.main_path = main_path
        self.data_path = data_path
        self.results_path = results_path
        self.subset = subset
        self.epochs = config.epochs
        self.saved_study = saved_study
        self.n_trials= n_trials

        self.get_datasets()
        utils.save_study_config(main_path = self.main_path,
                                results_path = self.results_path
                                )

    def get_datasets(self):
        self.train_dataset, self.test_dataset = utils.get_data(data_path = self.data_path,
                                                               subset = self.subset,
                                                               )
    def get_hyperparams(self, trial):
        self.hyperparams = Hyperparameters(trial).get_hyperparams()
    
    def objective(self, trial):
        trial_num = trial.number
        if trial_num >= self.n_trials-1:
            self.study.stop()
        
        self.trial_results = utils.get_results_dir(trial_num = trial_num,
                                                   results_path = self.results_path)
        self.get_hyperparams(trial = trial)
        training_data = DataLoader(self.train_dataset, 
                                   batch_size = self.hyperparams["batch_size"], 
                                   shuffle = False
                                   )
        testing_data = DataLoader(self.test_dataset, 
                                  batch_size = self.hyperparams["batch_size"],
                                  shuffle = False
                                  )
        if trial_num >= 1:
            self.study_dump()
        
        criterion = nn.CrossEntropyLoss()

        model = Network.CNN(num_conv_layers = self.hyperparams["num_conv_layers"],
                            in_channels = config.in_channels,
                            num_classes = config.num_classes
                            ).to(DEVICE)
        model.initialize_weights(initializer = self.hyperparams["initializer"])

        optimizer, scheduler = utils.get_optimizer_and_scheduler(model = model,
                                                                 optimizer_name = self.hyperparams["optimizer"],
                                                                 lr = self.hyperparams["lr"],
                                                                 scheduler_name = self.hyperparams["scheduler"],
                                                                 step_size = config.epochs // 5
                                                                 )
        print('\n', self.hyperparams)

        self.trainer = train_test.Trainer(trial = trial,
                                          trial_num = trial_num,
                                          train_data = training_data,
                                          test_data = testing_data,
                                          epochs = self.epochs,
                                          model = model,
                                          optimizer = optimizer,
                                          criterion = criterion,
                                          scheduler = scheduler,
                                          results_path = self.trial_results)
        
        best_acc = self.trainer.fit()
        torch.cuda.empty_cache()

        return best_acc

    def optimize(self):
        study_name = "MNIST_class_study"
        
        if self.saved_study is not None:
            self.study = joblib.load(self.saved_study)
            print("\nResume Study!")
        else:
            self.study = optuna.create_study(study_name = study_name,
                                             sampler = TPESampler(n_startup_trials = config.startup_trials),
                                             pruner = MedianPruner(n_startup_trials = config.startup_trials,
                                                                   n_warmup_steps = config.prune_warmup),
                                                                   direction = 'maximize'
                                                                   )
        self.study.optimize(self.objective, 
                            config.trials, 
                            gc_after_trial = True
                            )
        
        self.create_and_evaluate_summary()

    def study_dump(self):
        joblib.dump(self.study, os.path.join(self.results_path,
                                             "Optuna_Study.pkl"))
    def create_and_evaluate_summary(self):
        
        utils.save_and_evaluate_summary(self.study,
                                        self.results_path)
