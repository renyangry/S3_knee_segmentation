import torch
import torch.nn as nn



class MovPredictFNN(nn.Module):
    def __init__(self):
        super(MovPredictFNN, self, input_size, hidden_size, output_size).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size/2)
        self.fc3 = nn.Linear(hidden_size/2, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class MovPredictCNN1D(nn.Module):
    def __init__(self, input_size=23, hidden_size=64, output_size=5):
        super(MovPredictCNN1D, self).__init__()
        self.conv1d = nn.Conv1d(1, hidden_size, kernel_size=3)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(hidden_size * (input_size - 2), output_size)

    def forward(self, x):
        x = self.conv1d(x.unsqueeze(1))  # Add channel dimension
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
class MovPredictMLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MovPredictMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.Tanh()
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x
    





    