import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #print('Shape before flatten:', x.shape)
        x = x.view(x.size(0), -1)
        #print('Shape after flatten:', x.shape)
        #print('Shape fully connected:', self.fc1.shape)
        self.linear_input_size = x.shape[1]
        x = F.relu(nn.Linear(self.linear_input_size, 120)(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
