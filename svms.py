from sklearn import svm
import logging
import datetime

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
now = datetime.datetime.now().strftime('%m-%d@%H-%M')
handler = logging.FileHandler(f'logs/SVM_{now}.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:    |%(message)s')
handler.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(handler)
logger.addHandler(console)

import time


def log(msg='QAQ'):
    logger.info(str(msg))


class SVM(object):
    def __init__(self, data, label, inference_data, name='SVM', pca_dim=64 * 32):
        self.model = svm.SVC(gamma='scale', kernel='poly', degree=3, decision_function_shape='ovo')
        self.name = f'{name}scale-pca'
        t1 = time.time()
        self.preprocess(data, label, inference_data, pca_dim)
        print(f'time for pca: {(time.time()-t1):.1f}s')
        log(self.name)
        log(pca_dim)

    def ctrain(self):
        t1 = time.time()
        self.model.fit(self.train_data, self.train_label)
        print(f'time for train: {(time.time()-t1):.1f}s')
        train_acc = (self.model.predict(self.train_data) == self.train_label).sum() / len(self.train_label)
        print(f'train acc:{train_acc}')
        self.name = f'train{train_acc:.2f}-{self.name}'
        return

    def preprocess(self, data, label, inference_data, n_dim=64 * 16, val_split=0.9):
        from preprocess import by_pca
        import numpy as np

        data = data.reshape((data.shape[0], -1))
        inference_data = inference_data.reshape((inference_data.shape[0], -1))
        all_data = np.concatenate((data, inference_data))
        pca_all_data, pca_ratio = by_pca(all_data, n_dim)
        self.name = f'pcr{pca_ratio:.2f}-{self.name}'
        pca_data, pca_inference_data = pca_all_data[0:data.shape[0]], pca_all_data[data.shape[0]:]
        idx = np.random.permutation(len(label))
        data, label = pca_data[idx], label[idx]

        val_idx = int(val_split * len(label))
        self.train_data = data[val_idx:]
        self.train_label = label[val_idx:]
        self.val_data = data[:val_idx]
        self.val_label = label[:val_idx]
        self.inference_data = pca_inference_data

    def cvalidate(self):
        res = self.model.predict(self.val_data)
        hit = (res == self.val_label).sum()
        acc = hit / len(self.val_label)
        print(f'val acc: {acc:.4f} @[{hit}/{len(self.val_label)}]')
        self.name = f'val{acc:.2f}-{self.name}'
        return acc

    def cinference(self, data):
        print('inferencing')
        res = self.model.predict(self.inference_data)
        return res
