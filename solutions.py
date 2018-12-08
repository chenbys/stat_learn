import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models


class CNN(nn.Module):
    def __init__(self, class_num):
        super(CNN, self).__init__()
        self.class_num = class_num
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(3380, 50)
        self.fc2 = nn.Linear(50, class_num)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # [4,20,13,13],20*13*13=3380
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 3380)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def ctrain(self, train_data, train_label, val_data, val_label, lr=1e-4, batch_size=1, epoch_num=20):
        train_data = torch.tensor(train_data, device=self.device, dtype=torch.float32)
        train_label = torch.tensor(train_label, device=self.device, dtype=torch.long)
        val_data = torch.tensor(val_data, device=self.device, dtype=torch.float32)
        val_label = torch.tensor(val_label, device=self.device, dtype=torch.long)
        train_len, val_len = len(train_label), len(val_label)

        optimizer = optim.SGD(self.parameters(), lr=lr)
        for epoch_num in range(epoch_num):
            self.train()
            # train
            hit, total, loss_sum = 0, 0, 0
            for batch_idx in range(0, train_len, batch_size):
                data = train_data[batch_idx:batch_idx + batch_size]
                label = train_label[batch_idx:batch_idx + batch_size]
                optimizer.zero_grad()
                output = self.forward(data)
                loss = F.nll_loss(output, label)
                loss.backward()
                optimizer.step()

                loss_sum += loss.item()
                hit += (output.argmax(dim=1) == label).sum().item()
                total += len(label)
            print(f'train@ loss: {loss_sum/total:4.2f}, acc: {hit/total:4.2f}')

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
            print(f'valid@ loss: {loss_sum/total:4.2f}, acc: {hit/total:4.2f}')
            print()

    def ctest(self):
        pass

    def cinference(self):
        pass
