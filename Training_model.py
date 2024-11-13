's trimport torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn, optim

# Defining the Neural Network
class FashionNN(nn.Module):
    def __init__(self):
        super(FashionNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Preparing Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# Initializing Model, Loss Function, Optimizer
model = FashionNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
def train_model(model, train_loader, loss_fn, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        for batch, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            predictions = model(X)
            loss = loss_fn(predictions, y)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} completed.")

# Let's train and save the weights
train_model(model, train_loader, loss_fn, optimizer, epochs=10)
torch.save(model.state_dict(), 'fashion_mnist_model.pth')
print("Model weights saved to 'fashion_mnist_model.pth'")
