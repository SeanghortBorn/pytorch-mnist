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
        self.output = nn.Linear(32, 10)

    def forward(self, x):
        a2 = torch.sigmoid(self.layer2(x))
        a3 = torch.sigmoid(self.layer3(a2))
        return self.output(a3)

training_set = DataSet('D:\PROJECTS\PYTORCH\data\complete_mnist/training/', 10)

net = Net()
opt = optim.SGD(net.parameters(), lr=1)
cost_fn = nn.CrossEntropyLoss()

batch_size = 300
for iter in range(1000):
    opt.zero_grad()

    x, y = training_set.next_batch(batch_size)
    real_batch_size = len(x)
    x = torch.tensor(x, dtype=torch.float32)/255
    x = x.reshape(real_batch_size, -1)
    y = torch.tensor(y, dtype=torch.long)

    h = net(x)
    cost = cost_fn(h, y)

    if iter%10==0:
        print('[Epoch %02d] at iteration: %03d => Cost: %f' %(training_set.epochs_completed, iter, cost.item()))

    cost.backward()

    opt.step()

#evaluation
testing_set = training_set
n_correct = 0
testing_set.reset()
while testing_set.epochs_completed==0:
    x, y = testing_set.next_batch(batch_size, shuffle=False, equal_last_batch=False)
    real_batch_size = len(x)
    x = torch.tensor(x, dtype=torch.float32) / 255
    x = x.reshape(real_batch_size, -1)
    y = torch.tensor(y, dtype=torch.long)

    h = net(x)
    predicted = torch.argmax(h, dim=1)
    n_correct += (predicted == y).int().sum().item()
print("\nAccuracy: %.2f%%\n" % (n_correct * 100.0 / testing_set.num_examples))

# import torch
# import torch.nn as nn
# import torch.optim as optim
#
# from tools import img2tensor
# from dataset import DataSet
#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.layer2 = nn.Linear(784, 64)
#         self.layer3 = nn.Linear(64, 64)
#         self.layer4 = nn.Linear(64, 32)
#         self.out_layer = nn.Linear(32, 10)
#
#     def forward(self, x):
#         a2 = torch.sigmoid(self.layer2(x))
#         a3 = torch.sigmoid(self.layer3(a2))
#         a4 = torch.sigmoid(self.layer4(a3))
#         return self.out_layer(a4)
#
# train_set = DataSet('data/complete_mnist/training/', 10)
#
# net = Net()
# opt = optim.SGD(net.parameters(), lr=1)
# cost_fn = nn.CrossEntropyLoss()
#
# b_sz = 200
#
# for iter in range(1500):
#     opt.zero_grad()
#
#     x, y = train_set.next_batch(b_sz)
#     real_b_sz = len(x)
#     x = torch.tensor(x, dtype=torch.float32)/255 #tensor (b_sz, 28, 28)
#     x = x.reshape(real_b_sz, -1) #tensor (b_sz, 28*28)
#     y = torch.tensor(y, dtype=torch.long) #tensor (b_sz)
#
#     h = net(x) #tensor (b_sz, 3)
#     cost = cost_fn(h, y)
#
#     if iter%10==0:
#         print('[EPOCH %02d] At iteration: %03d => Cost: %f' %(train_set.epochs_completed, iter, cost.item()))
#
#     cost.backward()
#
#     opt.step()
#
# #evaluation
# test_set = train_set
# n_correct = 0
# test_set.reset()
# while test_set.epochs_completed==0:
#     x, y = test_set.next_batch(b_sz, shuffle=False, equal_last_batch=False)
#     real_b_sz = len(x)
#     x = torch.tensor(x, dtype=torch.float32) / 255  # tensor (b_sz, 28, 28)
#     x = x.reshape(real_b_sz, -1)  # tensor (b_sz, 28*28)
#     y = torch.tensor(y, dtype=torch.long)  # tensor (b_sz)
#
#     h = net(x) #tensor (m, 3)
#     predicted = torch.argmax(h, dim=1) #tensor (m)
#     n_correct += (predicted == y).int().sum().item()
# print("\nAccuracy: %.2f%%\n" % (n_correct * 100.0 / test_set.num_examples))