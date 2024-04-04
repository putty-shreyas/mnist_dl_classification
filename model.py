import torch.nn as nn
import torch.nn.init as init

class CNN(nn.Module):
    def __init__(self,
                 num_conv_layers,
                 in_channels,
                 num_classes
                 ):
        super(CNN, self).__init__()
        self.num_conv_layers = num_conv_layers
        
        # Define convolutional layers
        self.conv_layers = nn.ModuleList()
        for i in range(num_conv_layers):
            out_channels = 16 * 2**i
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1))
            self.conv_layers.append(nn.ReLU(inplace = True))
            self.conv_layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
            in_channels = out_channels
        
        # Calculate the size of the linear layer input
        self.final_width = 28 // (2 ** num_conv_layers)
        self.final_height = 28 // (2 ** num_conv_layers)
        self.linear_input_size = out_channels * self.final_width * self.final_height
        
        # Define fully connected layers
        self.fc1 = nn.Linear(self.linear_input_size, self.linear_input_size // 4)
        self.fc2 = nn.Linear(self.linear_input_size // 4, self.linear_input_size // 8)
        self.fc3 = nn.Linear(self.linear_input_size // 8, self.linear_input_size // 16)
        self.fc_out = nn.Linear(self.linear_input_size // 16, num_classes)
        
        # Define softmax layer
        self.softmax = nn.Softmax(dim = 1)

    def initialize_weights(self, initializer='kaiming_normal'):
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv2d):
                if initializer == 'kaiming_normal':
                    init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                elif initializer == 'xavier_normal':
                    init.xavier_normal_(layer.weight)
                elif initializer == 'uniform':
                    init.uniform_(layer.weight, -0.1, 0.1)
        
        for layer in [self.fc1, self.fc2, self.fc3, self.fc_out]:
            if isinstance(layer, nn.Linear):
                if initializer == 'kaiming_normal':
                    init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                elif initializer == 'xavier_normal':
                    init.xavier_normal_(layer.weight)
                elif initializer == 'uniform':
                    init.uniform_(layer.weight, -0.1, 0.1)

    def forward(self, x):
        # Apply convolutional layers
        for layer in self.conv_layers:
            x = layer(x)
        
        # Flatten the output
        x = x.view(-1, self.linear_input_size)

        # Apply softmax layer
        x = self.softmax(self.fc_out(self.fc3(self.fc2(self.fc1(x)))))
        
        return x
