import os
import torch
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

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

model = Net()
model.load_state_dict(torch.load('CNN.pth'))
model.eval()

cifar10_classes = ["Самолёт", "Автомобиль", "Птица", "Кошка", "Олень", "Собака", "Лягушка", "Лошадь", "Корабль", "Грузовик"]
folder_path = "C:\\Oleg\\Images\\CIFAR-10"

for filename in os.listdir(folder_path):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        # Загрузка изображения
        img = Image.open(os.path.join(folder_path, filename)).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        print('Класс от нейросети:', cifar10_classes[predicted.item()])

with torch.no_grad():
    output = model(img_tensor)
_, predicted = torch.max(output, 1)


