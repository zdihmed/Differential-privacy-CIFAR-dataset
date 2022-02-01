import torch
import torchvision
import matplotlib.pyplot as plt
from Net import Net
import torch.optim as optim
import torch.nn.functional as F
from TransformNormalDistribution import TransformNormalDistribution
import time
import numpy as np

import time

n_epochs = 5
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
normal_var = 0.25
normal_mean = 0
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


data_file = "../data/MNIST"
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(root=data_file, train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                   TransformNormalDistribution(normal_var, normal_mean)])),
    batch_size=batch_size_train, shuffle=True)

data_file = "../data/MNIST"
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(data_file, train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                   TransformNormalDistribution(normal_var, normal_mean)])),
    batch_size=batch_size_test, shuffle=True)


examples_training = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples_training)
print(example_data.size())
fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
fig
plt.show()


network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]
accuracy = []


def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            print()
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), 'results/model.pth')
            torch.save(optimizer.state_dict(), 'results/optimizer.pth')


def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    accuracy.append(100. * correct / len(test_loader.dataset))


for epoch in range(1, n_epochs + 1):
    train(epoch)
test()
