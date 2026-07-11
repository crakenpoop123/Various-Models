import torch
from torch import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 32 * 32
batch_size = 1000
num_classes = 10
num_epochs = 5
learning_rate = 0.01


# CIFAR10 dataset

train_dataset = torchvision.datasets.CIFAR10(
    root = "./data",
    train = True,
    transform = transforms.ToTensor(),
    download = True
)

test_dataset = torchvision.datasets.CIFAR10(
    root = "./data",
    train = False,
    transform = transforms.ToTensor(),
    download = True
)

# Data loader

train_loader = torch.utils.data.dataLoader(
    dataset = train_dataset,
    batch_size = batch_size,
    shuffle = True    
)

test_loader = torch.utils.data.dataLoader(
    dataset = test_dataset,
    batch_size = batch_size,
    shuffle = True    
)

# Neural Network init variables

conv0_out = 16
conv1_out = 48
conv2_out = 128

linear_input = 8 * 8
hidden_size = 4 * 4

hidden_depth = 2

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).to(device)
        # Convolutions
        self.convs = [
            nn.Conv2d(in_channels=3, out_channels=conv0_out, stride=1),
            nn.Conv2d(in_channels=conv0_out, out_channels=conv1_out, stride=1),
            nn.Conv2d(in_channels=conv1_out, out_channels=conv2_out, stride=1)
        ]
        self.avgPool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.inputL = nn.Linear(linear_input * conv2_out, hidden_size * conv2_out)

        self.linear = []
        for i in range(hidden_depth):
            self.linear.append(nn.Linear(hidden_size * conv2_out, hidden_size * conv2_out))
        
        self.outputL = nn.Linear(hidden_size * conv2_out, num_classes)
        self.relu = nn.ReLU()

        # Convolution block
    def conv_block(self, block, input):
        input = self.convs[block]
        input = F.relu(input)
        input = self.avgPool(input)
        return input
    
    def forward(self, x):
        # Convolutional blocks
        for i in range(len(self.convs)):
            x = self.conv_block(i, x)
        
        # Format the tensors
        x.view(-1, 8 * 8 * conv2_out)

        # Linear layers
        for i in range(len(self.linear)):
            x = self.linear[i](x)
        
        # Output layer
        x = self.outputL(x)
        x = self.relu(x)

        return x

