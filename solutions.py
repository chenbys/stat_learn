import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models


class CNN(nn.Module):
    def __init__(self, class_num=12, name='cnn'):
        super(CNN, self).__init__()
        self.class_num = class_num
        self.name = name

        self.fc_p = 0.5
        self.add_layers()

    def add_layers(self):
        self.num_for_fc = 3380
        self.conv1 = nn.Conv2d(1, 12, kernel_size=5)
        self.conv2 = nn.Conv2d(12, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(self.num_for_fc, 50)
        self.fc2 = nn.Linear(50, self.class_num)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.to(self.device)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # [4,20,13,13],20*13*13=3380
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, self.num_for_fc)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.fc_p, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def ctrain(self, train_data, train_label, val_data, val_label, lr=1e-5, batch_size=4, epoch_num=50, shuffle=True):
        train_data = torch.tensor(train_data, device=self.device, dtype=torch.float32)
        train_label = torch.tensor(train_label, device=self.device, dtype=torch.long)
        val_data = torch.tensor(val_data, device=self.device, dtype=torch.float32)
        val_label = torch.tensor(val_label, device=self.device, dtype=torch.long)
        train_len, val_len = len(train_label), len(val_label)

        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-2)
        train_acc, train_loss, val_acc, val_loss = [], [], [], []
        import time
        for epoch in range(epoch_num):
            start = time.time()
            self.train()
            # train
            if shuffle:
                import numpy as np
                idx = np.random.permutation(len(train_label))
                strain_data, strain_label = train_data[idx], train_label[idx]
            else:
                strain_data, strain_label = train_data, train_label

            hit, total, loss_sum = 0, 0, 0
            for batch_idx in range(0, train_len, batch_size):
                data = strain_data[batch_idx:batch_idx + batch_size]
                label = strain_label[batch_idx:batch_idx + batch_size]
                optimizer.zero_grad()
                output = self.forward(data)
                loss = F.nll_loss(output, label)
                loss.backward()
                optimizer.step()

                loss_sum += loss.item()
                hit += (output.argmax(dim=1) == label).sum().item()
                total += len(label)
            print(f'\nepoch: {epoch}')
            print(f'train@ loss: {loss_sum/total:4.4f}, acc: {hit/total:4.4f}')
            train_acc.append(hit / total)
            train_loss.append(loss_sum / total)

            # valdate
            self.eval()
            hit, total, loss_sum = 0, 0, 0
            for batch_idx in range(0, val_len, batch_size):
                data = val_data[batch_idx:batch_idx + batch_size]
                label = val_label[batch_idx:batch_idx + batch_size]
                output = self.forward(data)
                loss = F.nll_loss(output, label)

                loss_sum += loss.item()
                hit += (output.argmax(dim=1) == label).sum().item()
                total += len(label)
            print(f'valid@ loss: {loss_sum/total:4.4f}, acc: {hit/total:4.4f}')
            print(f'valid num: {val_len}')
            val_acc.append(hit / total)
            val_loss.append(loss_sum / total)
            over = time.time()
            print(f'timec@ {epoch}: {over-start:4.4f}s')

        self.train_acc = train_acc
        self.train_loss = train_loss
        self.val_acc = val_acc
        self.val_loss = val_loss

    def csave(self, prefix='cc'):
        torch.save(self.state_dict(),
                   f'params/{self.name}-{prefix}-{self.train_loss[-1]:.3f}-{self.val_loss[-1]:.3f}.pkl')

    def cload(self, lname='cnn'):
        self.load_state_dict(torch.load(f'params/{lname}.pkl'))

    def ctest(self):
        pass

    def cinference(self, data):
        self.eval()
        data = torch.tensor(data, device=self.device, dtype=torch.float32)
        output = self.forward(data)
        return output.argmax(dim=1)


class SCNN(CNN):
    def __init__(self, class_num=12, name='scnn'):
        super(SCNN, self).__init__(class_num, name)

    def add_layers(self):
        self.num_for_fc = 3920
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(self.num_for_fc, 30)
        self.fc2 = nn.Linear(30, self.class_num)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('in SCNN')
        print(self.device)
        self.to(self.device)


class SSCNN(CNN):
    def __init__(self, class_num=12, name='smscnn2'):
        super(SSCNN, self).__init__(class_num, name)

    def add_layers(self):
        self.num_for_fc = 1176
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3)
        self.conv2 = nn.Conv2d(6, 6, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(self.num_for_fc, 12)
        self.fc2 = nn.Linear(12, self.class_num)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('in SSCNN')
        print(self.device)
        self.to(self.device)


class SDropoutCNN(CNN):
    def __init__(self, class_num=12, name='LCNN'):
        super(SDropoutCNN, self).__init__(class_num, name)

    def add_layers(self):
        self.num_for_fc = 1176
        self.conv1 = nn.Conv2d(1, 30, kernel_size=3)
        self.conv2 = nn.Conv2d(30, 12, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(self.num_for_fc, 12)
        self.fc2 = nn.Linear(12, self.class_num)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc_p = 0.5
        print('in SDropoutCNN')
        print(self.device)
        self.to(self.device)
