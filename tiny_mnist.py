import torch
import torch.nn as nn
import torch.optim as optim

from tools import img2tensor
from dataset import DataSet


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer2 = nn.Linear(784, 64)
        self.layer3 = nn.Linear(64, 32)
        self.out_layer = nn.Linear(32, 3)

    def forward(self, x):
        a2 = torch.sigmoid(self.layer2(x))
        a3 = torch.sigmoid(self.layer3(a2))
        return self.out_layer(a3)


train_set = DataSet('data/tiny_mnist/', 3)

net = Net()
opt = optim.SGD(net.parameters(), lr=1)
cost_fn = nn.CrossEntropyLoss()

b_sz = 10

for iter in range(1000):
    opt.zero_grad()

    x, y = train_set.next_batch(b_sz)
    real_b_sz = len(x)
    x = torch.tensor(x, dtype=torch.float32) / 255  # tensor (b_sz, 28, 28)
    x = x.reshape(real_b_sz, -1)  # tensor (b_sz, 28*28)
    y = torch.tensor(y, dtype=torch.long)  # tensor (b_sz)

    h = net(x)  # tensor (b_sz, 3)
    cost = cost_fn(h, y)

    if iter % 10 == 0:
        print('[EPOCH %02d] At iteration: %03d => Cost: %f' % (train_set.epochs_completed, iter, cost.item()))

    cost.backward()

    opt.step()

# evaluation
test_set = train_set
n_correct = 0
test_set.reset()
while test_set.epochs_completed == 0:
    x, y = test_set.next_batch(b_sz, shuffle=False, equal_last_batch=False)
    real_b_sz = len(x)
    x = torch.tensor(x, dtype=torch.float32) / 255  # tensor (b_sz, 28, 28)
    x = x.reshape(real_b_sz, -1)  # tensor (b_sz, 28*28)
    y = torch.tensor(y, dtype=torch.long)  # tensor (b_sz)

    h = net(x)  # tensor (m, 3)
    predicted = torch.argmax(h, dim=1)  # tensor (m)
    n_correct += (predicted == y).int().sum().item()
print("\nAccuracy: %.2f%%\n" % (n_correct * 100.0 / test_set.num_examples))