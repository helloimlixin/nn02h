# A simple neural network for demo
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.utils.data import random_split


class SimpleNet(nn.Module):
    def __init__(self, in_features, n_classes):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(in_features, 50)
        self.fc2 = nn.Linear(50, n_classes)

        self.net = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2
        )

    def forward(self, x):
        return self.net(x)

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
in_features = 28*28
n_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5

# Load Data
train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_set, val_set = random_split(train_dataset, [55000, 5000])
test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Initialize network
model = SimpleNet(in_features=in_features, n_classes=n_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Get to correct shape
        data = data.reshape(data.shape[0], -1)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()



# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct/num_samples

# Print accuracy
model.eval()
model.to(device)
print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f} %")
print(f"Accuracy on validation set: {check_accuracy(val_loader, model)*100:.2f} %")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f} %")
