import torch
from torch import torchvision
import torchvision.transforms as transforms
import neuralnet


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


class NeuralNetwork(nn.Module):
    def __init__():
        l1 = 