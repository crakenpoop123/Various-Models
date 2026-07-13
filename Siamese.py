import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 32 * 32
batch_size = 1000
embed_dim = 16
num_epochs = 25
learning_rate = 0.005
gradient_accumulator = 2

# CIFAR10 dataset

train_dataset = torchvision.datasets.CIFAR10(
    root = "./data",
    train = True,
    transform = transforms.ToTensor(),
    download = True
)

# Data loader

train_loader = torch.utils.data.DataLoader(
    dataset = train_dataset,
    batch_size = batch_size,
    shuffle = True    
)

# Used for graphing the model's performance

step_arr = []
MSE_arr = []

step_diff_arr = []
MSE_diff_arr = []

saved_dot_avg = []

# Corruption variables

corruption = 0.1 # Proportion of pixels to be corrupted
max_pixel_corruption = 0.2 # Maximum intensity of corruption (original pixel value * (1 + max_pixel_corruption)). 
# Note: It has a different corruption value for each x, y, red, green, and blue, value of an image. 
# Also a pixel could have a corrupted red value but not green and blue

# Neural Network init variables

conv0_out = 9
conv1_out = 27
conv2_out = 81


linear_input = 4 * 4
hidden_size = 6 * 6
hidden_depth = 0

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
        
        self.intermediary = nn.Linear(hidden_size * conv2_out, int(embed_dim / 2))

        self.outputL = nn.Linear(int(embed_dim / 2), embed_dim)

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

        # print(x.size())

        x = self.inputL(x)

        # Linear layers
        for i in range(len(self.linear)):
            x = self.linear[i](x)
        
        # Output layer
        x = self.outputL(self.intermediary(x))

        return x


def train():
    print("Started training")

    n_total_steps = len(train_loader)

    model.zero_grad()

    same_img_pairwise_example = torch.zeros(embed_dim, embed_dim).to(device).fill_diagonal_(1)
    diff_img_pairwise_example = torch.zeros(embed_dim, embed_dim).to(device)

    for epoch in range(num_epochs):
        print("epoch: ", epoch + 1)

        for i, (images, labels) in enumerate(train_loader):

            # print(i)

            images = images.to(device)
            labels = labels.to(device)
            

            if i % gradient_accumulator < gradient_accumulator / 2:
                corrupted_images = corrupt(images)

                # Forward pass
                normal_output = F.normalize(model(images), dim=1)
                # with torch.no_grad():
                corrupted_output = F.normalize(model(corrupted_images), dim=1)
                

                # print(normal_output.size())
                # print(corrupted_output.size())
                dot_pairwise = torch.bmm(normal_output.unsqueeze(2), corrupted_output.unsqueeze(1)).to(device)

                # print(dot_pairwise.size())

                dot_avg_pairwise = dot_pairwise.mean(dim=0).to(device)

                if i == 0:
                    saved_dot_avg.append(dot_avg_pairwise.clone().detach().cpu().numpy())

                # print(dot_avg_pairwise.size())
                # print(dot_avg_pairwise)

                # Loss
                loss = criterion(dot_avg_pairwise, same_img_pairwise_example) * 10000
                prev_images = images
            else:
                corrupted_images = corrupt(prev_images)

                # Forward pass
                normal_output = F.normalize(model(images), dim=1)
                # with torch.no_grad():
                corrupted_output = F.normalize(model(corrupted_images), dim=1)
                

                # print(normal_output.size())
                # print(corrupted_output.size())
                dot_pairwise = torch.bmm(normal_output.unsqueeze(2), corrupted_output.unsqueeze(1)).to(device)

                # print(dot_pairwise.size())

                dot_avg_pairwise = dot_pairwise.mean(dim=0).to(device)

                if i == gradient_accumulator/2:
                    saved_dot_avg.append(dot_avg_pairwise.clone().detach().cpu().numpy())

                # print(dot_avg_pairwise.size())
                # print(dot_avg_pairwise)

                # Loss
                loss = criterion(dot_avg_pairwise, diff_img_pairwise_example) * 10000

            # Backward
            (loss / gradient_accumulator).backward()

            # Optimize
            if i % gradient_accumulator == int(gradient_accumulator/2 - 0.5): 
                step = epoch * n_total_steps + i

                optimizer.step()
                model.zero_grad()

                print("Similarity: Step: ", i, "/", n_total_steps, "MSE: ", loss.item())

                step_arr.append(step)
                MSE_arr.append(loss.item())
            elif i % gradient_accumulator == gradient_accumulator - 1: 
                step = epoch * n_total_steps + i

                optimizer.step()
                model.zero_grad()

                print("Difference: Step: ", i - 1, "/", n_total_steps, "MSE: ", loss.item())

                step_diff_arr.append(step - 1)
                MSE_diff_arr.append(loss.item())
    
                


def corrupt(input_image):
    images = input_image.clone().detach().to(device)
    
    # Randomize values for corruption
    rand_vals = torch.rand(images.shape, device=device) * max_pixel_corruption * 2 + 1 - 1 * max_pixel_corruption

    # Decide which values should be corrupted
    rand_vals_mask = torch.rand(images.shape, device=device) < corruption

    # Edit the image
    images[rand_vals_mask] *= rand_vals[rand_vals_mask]
    images /= (max_pixel_corruption + 1)

    return images


if __name__ == '__main__':
    # Init the model
    model = NeuralNetwork().to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Criterion. Mostly isn't used, only for a small bit to align the criterion of a different metric
    criterion = nn.MSELoss().to(device)

    train()



    # Save the model to a file

    PATH = "./Siamese.pth"

    torch.save(model, PATH)


    plt.figure(1)
    plt.title("Set A (same image)")
    print(len(saved_dot_avg))
    for i in range(int(len(saved_dot_avg) / 2)):
        plt.subplot(5, 5, i + 1)
        plt.imshow(saved_dot_avg[i * 2], cmap="gray")

    plt.figure(2)
    plt.title("Set B (different image)")
    for i in range(int(len(saved_dot_avg) / 2)):
        plt.subplot(5, 5, i + 1)
        plt.imshow(saved_dot_avg[i * 2 + 1], cmap="gray")


    plt.figure(3)
    plt.plot(step_arr, MSE_arr, color="black", marker="o", linestyle="--")
    plt.xlabel("Step")
    plt.ylabel("MSE")
    plt.title("Model's similarity error(comparing corrupted versions of the same image)")

    plt.figure(4)
    plt.plot(step_diff_arr, MSE_diff_arr, color="black", marker="o", linestyle="--")
    plt.xlabel("Step")
    plt.ylabel("MSE")
    plt.title("Model's difference error(Comparing different images)")

    plt.show()




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