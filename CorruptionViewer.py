import Siamese
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 1
num_imgs = 6

# Data

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

images = []
corrupted_images = []


def get_imgs():
    for i, (image, labels) in enumerate(test_loader):
        if i >= num_imgs:
            return f"Finished after {num_imgs} images"
        
        image = images.to(device)
        corrupted_image = Siamese.corrupt(image)

        images.append(image.permute(1, 2, 0))
        corrupted_images.append(corrupted_image.permute(1, 2, 0))

    
    return "Finished due to lack of data"

if __name__ == '__main__':
    get_imgs()


    # Show the images
    plt.figure(1)
    plt.title("Normal Images")
    for i in range(num_imgs):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i])

    plt.figure(2)
    plt.title("Corrupted Images")
    for i in range(num_imgs):
        plt.subplot(2, 3, i + 1)
        plt.imshow(corrupted_images[i])
    
    plt.show()