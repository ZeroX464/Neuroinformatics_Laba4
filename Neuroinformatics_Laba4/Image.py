import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageQt
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([transforms.ToTensor()])
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=True)

def convertToTensor(image):
    scaled_image = image.scaled(32, 32)
    pil_image = ImageQt.fromqimage(scaled_image)
    transform = transforms.Compose([transforms.PILToTensor()])
    img_tensor = transform(pil_image)
    return img_tensor

# Функция для сохранения изображений из DataLoader
def save_images_from_loader(loader, save_folder):
    for i, data in enumerate(loader):
        images, labels = data
        for j in range(len(images)):
            image = transforms.ToPILImage()(images[j])
            image.save(save_folder + 'image_' + str(i*len(images) + j) + '.png')
            
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

save_folder = "C:\\Oleg\\Images\\CIFAR-10"
#save_images_from_loader(test_loader, save_folder)