## MNIST Dataset Classification using tunable CNN architecture

Image Classification task performed on MNIST Dataset using CNN architecture. Purpose of the project was to show a clean and efficient Python-based Deep Learning project with hyperparameter optimization and tuning setup using Optuna

## Features 
 - Tunable CNN architecture achieving 98.29% accuracy
 - Hyperparameter tuning through Optuna and effective results storing

## Results and Summary 
<!-- Plot 1 --> 
<div>
    <img src="https://github.com/putty-shreyas/mnist_dl_classification/blob/main/Results/2024-04-03_13-32-38/Trial_19/Accuracy_Plot.png" alt="Plot 1" width="500" />
    <p>Accuracy Plot.</p>
</div>

<!-- Plot 2 -->
<div>
    <img src="https://github.com/putty-shreyas/mnist_dl_classification/blob/main/Results/2024-04-03_13-32-38/Trial_19/Loss_Plot.png" alt="Plot 2" width="500" />
    <p>Loss Plot.</p>
</div>

<!-- Plot 3 -->
<div>
    <img src="https://github.com/putty-shreyas/mnist_dl_classification/blob/main/Results/2024-04-03_13-32-38/Trial_19/Learning_Rate.png" alt="Plot 3" width="500" />
    <p>Learning Rate.</p>
</div>

<!-- Plot 4 -->
<div>
    <img src="https://github.com/putty-shreyas/mnist_dl_classification/blob/main/Results/2024-04-03_13-32-38/evaluated_summary.png" alt="Plot 4" width="1200" />
    <p>Final Study Evaluation.</p>
</div>

Summary:
 - In the Final Study Evaluation, the hyperparameter choices of the different trials are showcased and it can be seen that Optuna gave a certain specific set of Hyperparameters which gave similar, repeatable and stable results.
 - The accuracy, loss and learning rate plots show the training performance for the best Trial (Trial 19) in the Optuna study.
 - The important aspect of the results is that the losses and accuracy stabilise and merge at the end of the training and show results as expected.
 - All trial results are stored in their respective trials folders and we also can track training time through Trial summary file in results.
 - Models are saved at the best performing epochs to be used for validation later. 

## Getting Started
 - Run the run_main.py file in your environment and download the relevant packages if missing.
 - Possible packages that might need to be downloaded are openpyxl and Optuna.
 - You would also need to create folders "Data" and "Results" to download the data and store the results.
```
run run_main.py
```

## About Me
I am Shreyas Putty, a M.Sc. Graduate in Data Science and Machine Learning and I am passionate about finding creative solutions through my knowledge and skills. I have 3+ years of experience in Python and am open to any new opportunities.

## Contact
We can connect through my email id - putty.shreyas@gmail.com and through my Linkedin - https://www.linkedin.com/in/shreyas-subhash-putty/
