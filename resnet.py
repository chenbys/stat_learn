import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import datetime

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
now = datetime.datetime.now().strftime('%m-%d@%H-%M')
handler = logging.FileHandler(f'logs/resnet18_{now}.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:    |%(message)s')
handler.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(handler)
logger.addHandler(console)


def log(msg='QAQ'):
    logger.info(str(msg))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=12, name='resn18'):
        super(ResNet, self).__init__()
        self.name = name
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.linear = nn.Linear(512 * 4, num_classes)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(self.device)
        self.to(self.device)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def load_pretrained(self):
        import torchvision.models as models
        pretrained = models.resnet18(pretrained=True)
        pretrained_dict = pretrained.state_dict()
        model_dict = self.state_dict()
        update_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(update_dict)
        self.load_state_dict(model_dict)

    def ctrain(self, train_data, train_label, val_data, val_label, lr=1e-5, batch_size=4, epoch_num=50, shuffle=True):
        train_data = torch.tensor(train_data, device=self.device, dtype=torch.float32)
        train_label = torch.tensor(train_label, device=self.device, dtype=torch.long)
        val_data = torch.tensor(val_data, device=self.device, dtype=torch.float32)
        val_label = torch.tensor(val_label, device=self.device, dtype=torch.long)
        train_len, val_len = len(train_label), len(val_label)

        # optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-2)
        optimizer = optim.Adam(self.parameters(), amsgrad=True, weight_decay=1e-1)

        self.train_acc, self.train_loss, self.val_acc, self.val_loss = [], [], [], []
        min_loss = 0.09
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
            log(f'\nepoch: {epoch}')
            log(f'train@ loss: {loss_sum/total:4.4f}, acc: {hit/total:4.4f}')
            self.train_acc.append(hit / total)
            self.train_loss.append(loss_sum / total)

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
            log(f'valid@ loss: {loss_sum/total:4.4f}, acc: {hit/total:4.4f}')
            log(f'valid num: {val_len}')
            self.val_acc.append(hit / total)
            valloss = loss_sum / total
            self.val_loss.append(valloss)
            over = time.time()
            log(f'timec@ {epoch}: {over-start:4.4f}s')

            if valloss < min_loss:
                min_loss = valloss
                self.csave(f'{epoch}')

    def cinference(self, data):
        self.eval()
        length = len(data)
        batch_size = 600
        res = []
        for i in range(0, length, batch_size):
            d = data[i:i + batch_size]
            d = torch.tensor(d, device=self.device, dtype=torch.float32)
            output = self.forward(d)
            r = output.argmax(dim=1)
            res += r.tolist()
        return res

    def csave(self, prefix='cc'):
        name = f'params/{self.name}-{prefix}-{self.train_loss[-1]:.3f}-{self.val_loss[-1]:.3f}.pkl'
        log(name)
        torch.save(self.state_dict(), name)

    def cload(self, lname='cnn'):
        self.load_state_dict(torch.load(f'params/{lname}.pkl'))


def ResNet18(name='res18'):
    return ResNet(BasicBlock, [2, 2, 2, 2], name=name)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    log(y.size())

# test()
