import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import datetime
import math

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


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=12, name='res18'):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(4, stride=2)
        self.fc = nn.Linear(512 * 9, num_classes)

        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(self.device)
        self.to(self.device)
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

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
