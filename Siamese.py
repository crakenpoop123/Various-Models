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

corruption = 0.3 # Proportion of pixels to be corrupted
max_pixel_corruption = 0.5 # Maximum intensity of corruption (original pixel value * (1 + max_pixel_corruption)). 
# Note: It has a different corruption value for each x, y, red, green, and blue, value of an image. 
# Also a pixel could have a corrupted red value but not green and blue

linear_input = 4 * 4
hidden_size = 4 * 4

hidden_depth = 2

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Convolutions
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=3, out_channels=conv0_out, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=conv0_out, out_channels=conv1_out, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=conv1_out, out_channels=conv2_out, kernel_size=3, stride=1, padding=1)
        ])
        self.avgPool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.inputL = nn.Linear(linear_input * conv2_out, hidden_size * conv2_out)

        self.linear = nn.ModuleList([])
        for i in range(hidden_depth):
            self.linear.append(nn.Linear(hidden_size * conv2_out, hidden_size * conv2_out))
        
        self.outputL = nn.Linear(hidden_size * conv2_out, embed_dim)

        # Convolution block
    def conv_block(self, block, input):
        input = self.convs[block](input)
        input = F.relu(input)
        input = self.avgPool(input)
        return input
    
    def forward(self, x):
        # Convolutional blocks
        for i in range(len(self.convs)):
            x = self.conv_block(i, x)
        
        # Format the tensors
        x = x.view(-1, linear_input * conv2_out)

        # Linear layers
        for i in range(len(self.linear)):
            x = self.linear[i](x)
        
        # Output layer
        x = self.outputL(x)

        return x

def train():
    print("Started training")

    n_total_steps = len(train_loader)

    model.zero_grad()

    for epoch in range(num_epochs):
        print("epoch: ", epoch + 1)

        for i, (images, labels) in enumerate(train_loader):

            # print(i)

            images = images.to(device)
            labels = labels.to(device)

            corrupted_images = corrupt(corruption, images)

            # Forward pass
            normal_output = model(images)
            with torch.no_grad():
                corrupted_output = model(corrupted_images)
            
            # Loss
            loss = criterion(normal_output, corrupted_output)

            # Backward
            loss.backward()

            # Optimize

            if i% 10 == 0: 
                optimizer.step()
                model.zero_grad()
                print("Step: ", i, "/", n_total_steps, ". MSE: ", loss.item())





def corrupt(intensity, input_image):
    images = input_image.clone().detach().to(device)
    
    # Randomize values for corruption
    rand_vals = torch.rand(images.shape, device=device) * max_pixel_corruption * 2 + 1 - 1 * max_pixel_corruption

    # Decide which values should be corrupted
    rand_vals_mask = torch.rand(images.shape, device=device) < intensity

    # Edit the image
    images[rand_vals_mask] *= rand_vals[rand_vals_mask]
    images /= (max_pixel_corruption + 1)

    return images


if __name__ == '__main__':
    # Init the model
    model = NeuralNetwork().to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Criterion. This will be replaced later by a better method that lowers feature collaps
    criterion = nn.MSELoss().to(device)

    train()

# This went in the train loop but is not currently used and was taking up many lines

            # corrupted_image = corrupt(corruption, images[0]).permute(1, 2, 0)

            # View the corrupted and non-corrupted images

            # curr_image = images[0].clone().detach().permute(1, 2, 0)

            # plt.subplot(2, 3, 1)
            # plt.figure(1)
            # plt.imshow(curr_image.cpu().numpy())

            # # plt.subplot(2, 3, 2)
            # plt.figure(2)
            # plt.imshow(corrupted_image.cpu().numpy()) 

            # plt.show()