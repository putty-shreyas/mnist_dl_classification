import torch
subset = False
if subset:
    high = 2000

num_classes = 10
in_channels = 1

startup_trials = 10
prune_warmup = 10
trials = 20
epochs = 20

#Set Computation Device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")