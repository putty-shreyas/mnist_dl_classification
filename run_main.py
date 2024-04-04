import datetime
from pathlib import Path
import os

import config
import optuna_optimizer

import warnings
warnings.filterwarnings("ignore")

date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

main_path = Path(__file__).parents[0].__str__()

data_path = os.path.join(main_path, "Data")
results_path = os.path.join(main_path, "Results", f"{date}")

saved_study = None #r

tuner = optuna_optimizer.OptunaOptimizer(main_path = main_path,
                                         data_path = data_path,
                                         results_path = results_path,
                                         subset = config.subset,
                                         saved_study = saved_study,
                                         n_trials = config.trials
                                         )
tuner.optimize()