import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import datasets, transforms

# Defining a custom dataset class
class FashionMNISTCustom(Dataset):
    def __init__(self, url, transform=None):
        self.data = datasets.FashionMNIST(root='./data', train=True if 'train' in url else False, download=True, transform=transform)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

from torchvision import transforms

# transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Loading datasets directly
train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Creating data loaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

import torch.nn as nn
import torch.nn.functional as F

class FashionNN(nn.Module):
    def __init__(self):
        super(FashionNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initializing model
model = FashionNN()

import torch.optim as optim

# Hyperparameters
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# Training loop
def train(model, train_loader, loss_fn, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        for batch, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            if batch % 10 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Train the model
train(model, train_loader, loss_fn, optimizer)

# Save the model weights
torch.save(model.state_dict(), 'fashion_mnist_model.pth')

# Load the saved model
model = FashionNN()
model.load_state_dict(torch.load('fashion_mnist_model.pth'))
model.eval()

# Evaluation
def evaluate(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in test_loader:
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")

evaluate(model, test_loader)
