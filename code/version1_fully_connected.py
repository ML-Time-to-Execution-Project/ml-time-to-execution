# neural network
# -----------------------------------------------------

import math
import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import argparse
import datetime

# time start
start = datetime.datetime.now()

# parameter setting
parser = argparse.ArgumentParser(description='hyper_parameter')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--hidden_size1', type=int, default=50, help='the number of first layer')
parser.add_argument('--hidden_size2', type=int, default=20, help='the number of second layer')
parser.add_argument('--censor_weight', type=float, default=0.1, help='weight of censor loss')
parser.add_argument('--time_weight', type=float, default=1, help='weight of time loss')
parser.add_argument('--num_epochs', type=int, default=50, help='the number of epochs')
args = parser.parse_args()

learning_rate = args.lr
hidden_size1 = args.hidden_size1
hidden_size2 = args.hidden_size1
censor_weight = args.censor_weight
time_weight = args.time_weight
num_epochs = args.num_epochs
input_size = 102

print(f'learning rate = {learning_rate}, hidden_size1 = {hidden_size1}, hidden_size2 = {hidden_size2}'
      f'censor weight = {censor_weight}, time weight = {time_weight}, num epochs = {num_epochs}')

# data processing

sc_x = StandardScaler()
sc_time = StandardScaler()


class Standardized:
    def __call__(self, sample):
        inputs, time_label, censor_label = sample
        sample = sc_x.transform(inputs), sc_time.transform(time_label), censor_label
        return sample


class ToTensor:
    def __call__(self, sample):
        inputs, time_label, censor_label = sample
        return torch.from_numpy(inputs), torch.from_numpy(time_label), torch.from_numpy(censor_label)
        # return torch.Tensor(inputs), torch.Tensor(time_label), torch.Tensor(censor_label)


class OrderBookDataset(Dataset):
    def __init__(self, transform=None, path1=f'test_data/buy_x_train.csv',
                 path2=f'test_data/buy_y_train.csv'):
        # data loading
        x = np.loadtxt(path1, dtype=np.float32, delimiter=',')
        y = np.loadtxt(path2, dtype=np.float32, delimiter=',')

        # self.x = torch.from_numpy(x)
        # self.y_time = torch.unsqueeze(torch.from_numpy(y[:, 0]),dim=1)
        # self.y_censor = torch.unsqueeze(torch.from_numpy(y[:, 1]),dim=1)
        self.x = x
        self.y_time = np.log(y[:, 0]).reshape(-1, 1)
        self.y_censor = y[:, 1].reshape(-1, 1)
        self.n_samples = y.shape[0]
        self.transform = transform

    def __getitem__(self, index, ):
        sample = self.x[index].reshape(-1,102), self.y_time[index].reshape(-1,1), self.y_censor[index].reshape(-1, 1)
        if self.transform:
            sc_x.fit(self.x)
            sc_time.fit(self.y_time)
            sample = self.transform(sample)
        return sample
        # dataset

    def __len__(self):
        return self.n_samples


composed = torchvision.transforms.Compose([Standardized(), ToTensor()])
# composed = torchvision.transforms.Compose([ToTensor()])
dataset = OrderBookDataset(transform=composed)
dataloader = DataLoader(dataset=dataset, batch_size=30, shuffle=True, num_workers=0)


# model building
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_size=2):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size1)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.l3 = nn.Linear(hidden_size2, num_size)
        self.norm1 = nn.BatchNorm1d(input_size)
        self.norm2 = nn.BatchNorm1d(hidden_size1)
        self.norm3 = nn.BatchNorm1d(hidden_size2)
        nn.init.xavier_uniform_(self.l1.weight)
        nn.init.xavier_uniform_(self.l2.weight)
        nn.init.xavier_uniform_(self.l3.weight)

    def forward(self, x):
        out = self.norm1(x)
        out = self.l1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.l2(out)
        out = self.norm3(out)
        out = self.relu(out)
        out = self.l3(out)
        return out


# loss function
model = NeuralNet(input_size, hidden_size1, hidden_size2)
criterion1 = nn.MSELoss()
criterion2 = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# loop iteration
for epochs in range(num_epochs):
    for i, (inputs, time_label, censor_label) in enumerate(dataloader):
        # forward backward update
        inputs = torch.squeeze(inputs)
        # print(inputs.size())
        time_label = torch.squeeze(time_label, 1)
        # print(time_label.size())
        censor_label = torch.squeeze(censor_label, 1)
        # forward
        y_predicted = model(inputs)
        # print(y_predicted.size())
        out1 = torch.unsqueeze(y_predicted[:, 0], 1)
        # print(out1.size())
        out2 = torch.unsqueeze(y_predicted[:, 1], 1)

        # print(censor_label.size())
        # print(torch.unsqueeze(torch.sigmoid(out1), dim=1).size())
        # print(torch.unsqueeze(torch.sigmoid(out1), dim=1))
        # print(torch.unsqueeze(censor_label, dim=1))


        loss1 = criterion1(out1, time_label)
        loss2 = criterion2(out2, censor_label)

        # print(loss1.size())
        # print(loss2.size())

        # loss = criterion1(y_predicted, time_label)
        loss = (time_weight * loss1 + censor_weight * loss2)
        # print(loss.size())
        # backward pass
        loss.backward()

        # update
        optimizer.step()
        optimizer.zero_grad()

        if (i + 1) % 30 == 0:
            print(f'epoch:{epochs + 1}, loss = {loss.item():.4f}')

# test result
y_test = np.loadtxt('test_data/buy_y_test.csv', dtype=np.float32, delimiter=',')
x_test = np.loadtxt('test_data/buy_x_test.csv', dtype=np.float32, delimiter=',')
x_test = torch.from_numpy(sc_x.transform(x_test.reshape(-1, 102)))
y_time_test = torch.from_numpy(sc_time.transform(np.log(y_test[:, 0].reshape(-1, 1))))
y_censor_test = torch.from_numpy(y_test[:, 1].reshape(-1, 1))

with torch.no_grad():
    y_predicted = model(x_test)
    y_time_predicted = y_predicted[:, 0].reshape(-1, 1)
    y_censor_predicted = y_predicted[:, 1].reshape(-1, 1)
    y_predicted_cls = torch.sigmoid(y_censor_predicted).round()
    censor_acc = (y_predicted_cls == y_censor_test).sum() / float(y_censor_test.shape[0])
    time_loss = criterion1(y_time_predicted, y_time_test)
    # print(y_predicted_cls.eq(y_censor_test).sum())
    print(f'y time predicted = {y_time_predicted}')
    print(f'y time test = {y_time_test}')
    print(f'y censor predicted = {y_predicted_cls}')
    print(f'y censor test = {y_censor_test}')
    print(f'censor accuracy = {censor_acc:4f}, time loss = {time_loss:4f}')


# time end
end = datetime.datetime.now()
print('Running time: %s Seconds' % (end-start))
