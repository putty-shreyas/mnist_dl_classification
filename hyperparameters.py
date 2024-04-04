class Hyperparameters():
    def __init__(self, 
                 trial = None                 
                 ):
        self.trial = trial
        
    def training(self):
        batch_size = self.trial.suggest_categorical("Batch_size", [512, 1024, 2048])
        optimizer = self.trial.suggest_categorical("Optimizer", ["Adam", "SGD"])
        scheduler = self.trial.suggest_categorical("Scheduler", ["ExponentialLR", "StepLR"])
        lr = self.trial.suggest_loguniform("Learning_Rate",
                                           low = 1e-4,
                                           high = 1e-2
                                           )

        training_params = {"batch_size": batch_size,
                           "optimizer": optimizer,
                           "scheduler": scheduler,
                           "lr": lr}
        return training_params            
    
    def model(self):
        num_conv_layers = self.trial.suggest_int("Conv_Layers", 
                                                low = 1, 
                                                high = 3,
                                                step = 1
                                                )
        initializer = self.trial.suggest_categorical("Initializer", ["kaiming_normal", "xavier_normal", "uniform"])

        model_params = {"num_conv_layers": num_conv_layers,
                        "initializer": initializer}
        return model_params
    
    def get_hyperparams(self):
        training_params = self.training()
        model_params = self.model()
        return {**training_params, **model_params}
