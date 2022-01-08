#Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as T
#Create fully connected network
class NN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(in_channels, 512)
        self.fc2 = nn.Linear(512, out_channels)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
#Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Hyperparameters
in_channels = 784
out_channels = 10
epochs = 10
batch_size = 10
learning_rate = 1e-4
#Load data
train_dataset = datasets.MNIST(root="dataset/", train=True, transform=T.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root="dataset/", train=False, transform=T.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
#Initialize network
model = NN(in_channels=in_channels, out_channels=out_channels).to(device)
#Loss and optimizer
criteria = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
#Train network
for epoch in range(epochs):
    for ite, (data, targets) in enumerate(train_loader):
        #Get data if possible
        data = data.to(device=device)
        targets = targets.to(device=device)
        #Get to correct shape
        data = data.reshape(data.shape[0], -1)

        #forward
        score = model(data)
        loss = criteria(score, targets)

        #backward
        optimizer.zero_grad()
        loss.backward()

        #gradient descent or Adam step
        optimizer.step()

#Check accuracy on training & test see how good your model
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on testing data")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, prediction = score.max(1)
            num_correct += (prediction == y).sum()
            num_samples += prediction.size(0)

        print(f'Got {num_correct}/{num_samples} with accuracy {(float(num_correct)/float(num_samples))*100: .2f}')

    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)

























