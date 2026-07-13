import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from Siamese import NeuralNetwork
import Siamese

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


input_size = 32 * 32
batch_size = 1000
embed_dim = 16
num_epochs = 100
learning_rate = 0.005
gradient_accumulator = 2


test_dataset = torchvision.datasets.CIFAR10(
    root = "./data",
    train = False,
    transform = transforms.ToTensor(),
    download = True
)

# Data loader

test_loader = torch.utils.data.DataLoader(
    dataset = test_dataset,
    batch_size = batch_size,
    shuffle = True    
)


# Load the model

PATH = "./Siamese.pth"

loaded_model = torch.load(PATH, weights_only=False).to(device)

loaded_model.eval()


# This will be used to view the matrices

saved_dot_avg = []

with torch.no_grad():

    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)

        if i < 6:
            if i != 0:
                corrupted_diff_images = Siamese.corrupt(prev_images).to(device)
                corrupted_images = Siamese.corrupt(images).to(device)
                # Forward pass
                normal_output = F.normalize(loaded_model(images), dim=1).to(device)

                corrupted_output = F.normalize(loaded_model(corrupted_images), dim=1)
                corrupted_diff_output = F.normalize(loaded_model(corrupted_diff_images), dim=1)
                
                # print(normal_output.size())
                # print(corrupted_output.size())
                dot_pairwise = torch.bmm(normal_output.unsqueeze(2), corrupted_output.unsqueeze(1)).to(device)
                dot_diff_pairwise = torch.bmm(normal_output.unsqueeze(2), corrupted_diff_output.unsqueeze(1)).to(device)

                dot_diff_avg_pairwise = dot_diff_pairwise.mean(dim=0).to(device)
                dot_avg_pairwise = dot_pairwise.mean(dim=0).to(device)
                
                saved_dot_avg.append(dot_avg_pairwise.clone().detach().cpu().numpy())
                saved_dot_avg.append(dot_diff_avg_pairwise.clone().detach().cpu().numpy())
            prev_images = images
    
    plt.figure(1)
    for i in range(int(len(saved_dot_avg) / 2)):
        plt.subplot(2, 3, i + 1)
        plt.imshow(saved_dot_avg[i * 2], cmap="gray")
    
    plt.title("Collection of 6 same image dot matrices from the model")

    plt.figure(2)
    for i in range(int(len(saved_dot_avg) / 2)):
        plt.subplot(2, 3, i + 1)
        plt.imshow(saved_dot_avg[i * 2 + 1], cmap="gray")
    
    plt.title("Collection of 6 different image dot matrices from the model")
    plt.show()