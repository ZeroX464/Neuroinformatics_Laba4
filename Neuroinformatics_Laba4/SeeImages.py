import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 65)
        self.fc3 = nn.Linear(65, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
net.load_state_dict(torch.load("CNN.pth"))

cifar10_classes = ["Самолёт", "Автомобиль", "Птица", "Кошка", "Олень", "Собака", "Лягушка", "Лошадь", "Корабль", "Грузовик"]

net.eval()
with torch.no_grad():
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    print(labels)
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    print(predicted)
    print("Класс изображения - класс от нейросети:")
    for i in range(16):
        print("{} - {}".format(cifar10_classes[labels[i]], cifar10_classes[predicted[i]]))

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

imshow(torchvision.utils.make_grid(images))
