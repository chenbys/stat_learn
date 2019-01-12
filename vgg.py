'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
import datetime

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
now = datetime.datetime.now().strftime('%m-%d@%H-%M')
handler = logging.FileHandler(f'logs/vgg11_{now}.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:    |%(message)s')
handler.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(handler)
logger.addHandler(console)


def log(msg='QAQ'):
    logger.info(str(msg))


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, name='', vgg_name='VGG11'):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(2048, 12)
        self.name = name + vgg_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(self.device)
        self.to(self.device)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return F.log_softmax(out, dim=1)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True),
                           nn.Dropout2d(p=0.5)
                           ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def load_pretrained(self):
        import torchvision.models as models
        pretrained = models.vgg11_bn(pretrained=True)
        pretrained_dict = pretrained.state_dict()
        model_dict = self.state_dict()
        update_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(update_dict)
        self.load_state_dict(model_dict)

    def ctrain(self, train_data, train_label, val_data, val_label, lr=1e-3, wd=5e-4,
               batch_size=4, epoch_num=50, shuffle=True, val_split=0.5):
        info = f'lr{lr}_wd{wd}_bs{batch_size}_vs{val_split}'
        self.name = self.name + info
        log(self.name)
        train_data = torch.tensor(train_data, device=self.device, dtype=torch.float32)
        train_label = torch.tensor(train_label, device=self.device, dtype=torch.long)
        val_data = torch.tensor(val_data, device=self.device, dtype=torch.float32)
        val_label = torch.tensor(val_label, device=self.device, dtype=torch.long)
        train_len, val_len = len(train_label), len(val_label)

        # optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-2)
        optimizer = optim.Adam(self.parameters(), lr=lr, amsgrad=True, weight_decay=wd)

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
        name = f'params2/{self.name}-{prefix}-{self.train_loss[-1]:.3f}-{self.val_loss[-1]:.3f}.pkl'
        log(name)
        torch.save(self.state_dict(), name)

    def cload(self, lname='cnn'):
        self.load_state_dict(torch.load(f'params/{lname}.pkl'))
