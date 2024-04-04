import config
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import shutil
from torch.utils.data import Subset
from torchvision import datasets, transforms
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR

def save_study_config(main_path, results_path):
    main_config = os.path.join(main_path, "config.py")
    
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    results_config = os.path.join(results_path, "config.py")
    
    shutil.copyfile(main_config, results_config)

def get_results_dir(trial_num, results_path):
    trial_results_path = os.path.join(results_path, f"Trial_{trial_num}")
    
    if not os.path.exists(trial_results_path):
        os.makedirs(trial_results_path)
    
    return trial_results_path

def get_data(data_path, subset):
    # Define transformations to apply to the data
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to tensors
        transforms.RandomRotation(20),
        transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values to range [-1, 1]
    ])

    # Download MNIST dataset
    train_dataset = datasets.MNIST(root = data_path,
                                    train = True,
                                    transform = transform,
                                    download = True
                                    )
    test_dataset = datasets.MNIST(root = data_path,
                                    train = False,
                                    transform = transform,
                                    download = True
                                    )
    
    if subset:
        train_dataset = Subset(train_dataset, range(0, config.high))
        test_dataset = Subset(train_dataset, range(0, config.high // 2))
    
    print("Train Dataset Length: ", len(train_dataset))
    print("Test Dataset Length: ", len(test_dataset))

    return train_dataset, test_dataset

def get_optimizer_and_scheduler(model,
                                optimizer_name,
                                lr,
                                scheduler_name,
                                step_size
                                ):
    # Initialize optimizer
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f'Optimizer {optimizer_name} not supported.')
    
    # Initialize learning rate scheduler
    if scheduler_name == 'StepLR':
        scheduler = StepLR(optimizer, step_size = step_size, gamma = 0.9)
    elif scheduler_name == 'ExponentialLR':
        scheduler = ExponentialLR(optimizer, gamma = 0.995)
    else:
        raise ValueError(f'Scheduler {scheduler_name} not supported.')
    
    return optimizer, scheduler

def save_plot_lr(learning_rates, 
                 results_path,
                 ):
    
    plt.plot(learning_rates, label = "Learning Rate")
    plt.grid(True)
    plt.title("Change of Learning Rate over Training Period")
    plt.xlabel("Iterations")
    plt.ylabel("Learning Rate")
    plt.legend()
    
    plt.savefig(os.path.join(results_path, "Learning_Rate.png"), 
                            dpi = 200)
    plt.clf()   
    plt.close()

    train_lr = pd.DataFrame({"Learning Rate" : learning_rates})
    train_lr.index.name = "Iterations"
    train_lr.to_excel(os.path.join(results_path, f"Learning_Rate.xlsx"))

def save_metrics(loss_list,
                 accuracy_list,
                 results_path,
                 train: bool = True,
                 ):
    task = ["Train" if train else "Test"][0]

    metric_df = pd.DataFrame({f"{task}_Loss": loss_list,
                              f"{task}_Accuracy": accuracy_list
                            })
    metric_df.index.name = "Epochs"
    metric_df.to_excel(os.path.join(results_path, 
                                     f"{task}_metrics.xlsx"))

def plot_metrics(train_list,
                 test_list,
                 results_path,
                 loss: bool = True
                 ):
    metric = ["Loss" if loss else "Accuracy"][0]

    plt.figure(figsize = (8, 6))
    plt.plot(train_list, label = f'Train_{metric}')
    plt.plot(test_list, label = f"Test_{metric}")
    plt.grid(True)
    plt.legend()
    plt.ylabel(f'{metric}', size = 15)
    plt.xlabel('Epochs', size = 15)
    plt.title(f'{metric} Plot', size = 20)
    plt.savefig(os.path.join(results_path, f"{metric}_Plot.png"), 
                dpi = 200)
    plt.clf()
    plt.close()

def save_and_evaluate_summary(study, results_path):
    
    params_dict = study.best_trial.params.copy()
        
    best_stats = {**params_dict,
                  "Best_trial" : study.best_trial.number, 
                  "Accuracy" : study.best_trial.value * 100,
                  "Loss" : study.best_trial.user_attrs['Test_loss'],
                  "Epoch" : study.best_trial.user_attrs['Epoch']}
    
    with open(os.path.join(results_path, "Best_trial.txt"),
              "w") as fp:
        json.dump(best_stats, fp, indent = 2)
                    
    summary_df = study.trials_dataframe()
    summary_df.to_excel(os.path.join(results_path, "Trials_summary.xlsx")) 
    
    summary_df = summary_df.loc[summary_df["state"] == "COMPLETE"]
    summary_df = summary_df.drop(
                ["datetime_start", "datetime_complete", "duration"], axis=1)

    cols = summary_df.columns.tolist()
    pd.set_option('display.max_columns', None)
    cols[:] = [c.replace('params_', '') for c in cols]
    cols[:] = [c.replace('user_attrs_', '') for c in cols]
    cols = ['Trial' if c == 'number' else c for c in cols]
    cols = ['Accuracy' if c == 'value' else c for c in cols]
    
    summary_df.columns = cols
    summary_df["Accuracy"] = summary_df["Accuracy"].apply(lambda x: x * 100)
    
    summary_df.sort_values(by = 'Accuracy', 
                           inplace = True,
                           ascending = False
                           )

    summary_df.to_excel(os.path.join(results_path, "Trial_evaluated_summary.xlsx"))
    print("\nEvaluation saved successfully!")

