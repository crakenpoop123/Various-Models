import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 32 * 32
batch_size = 1000
embed_dim = 256
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

train_loader = torch.utils.data.DataLoader(
    dataset = train_dataset,
    batch_size = batch_size,
    shuffle = True    
)

test_loader = torch.utils.data.DataLoader(
    dataset = test_dataset,
    batch_size = batch_size,
    shuffle = True    
)

view = iter(test_loader)
view_data, view_targets = next(view)

# Neural Network init variables

conv0_out = 16
conv1_out = 48
conv2_out = 128

corruption = 1

linear_input = 8 * 8
hidden_size = 4 * 4

hidden_depth = 2

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Convolutions
        self.convs = [
            nn.Conv2d(in_channels=3, out_channels=conv0_out, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=conv0_out, out_channels=conv1_out, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=conv1_out, out_channels=conv2_out, kernel_size=3, stride=1)
        ]
        self.avgPool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.inputL = nn.Linear(linear_input * conv2_out, hidden_size * conv2_out)

        self.linear = []
        for i in range(hidden_depth):
            self.linear.append(nn.Linear(hidden_size * conv2_out, hidden_size * conv2_out))
        
        self.outputL = nn.Linear(hidden_size * conv2_out, embed_dim)
        self.softmax = nn.Softmax()

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
        x = self.softmax(x)

        return x

def train():
    print("Started training")

    for epoch in range(num_epochs):
        print("epoch: ", epoch + 1)

        for i, (images, labels) in enumerate(train_loader):

            print(i)

            images = images.to(device)
            labels = labels.to(device)


            corrupted_images = corrupt(corruption, images[0])
            corrupted_images = corrupted_images.permute(1, 2, 0)

            # View the corrupted and non-corrupted images

            curr_image = images[0].clone().detach().permute(1, 2, 0)

            # plt.subplot(2, 3, 1)
            plt.figure(1)
            plt.imshow(curr_image.cpu().numpy())

            # plt.subplot(2, 3, 2)
            plt.figure(2)
            plt.imshow(corrupted_images.cpu().numpy()) 

            plt.show()

def corrupt(intensity, input_image):
    image = input_image.clone().detach()
    for r, x in enumerate(image):
        for g, y in enumerate(x):
            for b, z in enumerate(y):
                # print(image.dtype)
                if random.random() < intensity:
                    # print("r: ", r, "g: ", g, "b: ", b)
                    image[r, g, b] *= ((random.random()) + 0.5) / 1.5
                    
                    # image[r, g, b] = torch.Tensor([0, 0, 0])
    
    return image


if __name__ == '__main__':
    # Init the model
    model = NeuralNetwork().to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train()
