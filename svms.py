from sklearn import svm
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


class SVM(object):
    def __init__(self, data, label, inference_data, name='SVM', pca_dim=64 * 20):
        self.model = svm.SVC(gamma='scale')
        self.name = f'{name}scale-raw'
        self.pca_preprocess(data, label, inference_data, n_dim=pca_dim)

    def ctrain(self):
        import time
        t1 = time.time()
        self.model.fit(self.train_data, self.train_label)
        t2 = time.time()
        log(f'train size:{len(self.train_label)}')
        train_acc = (self.model.predict(self.train_data) == self.train_label).sum() / len(self.train_label)
        log(f'train acc:{train_acc}')
        self.name = f'{self.name}-train{train_acc:.2f}@{(t2-t1)}s'

    def TS_train(self, mess_label):
        import numpy as np
        mess_train_data = np.concatenate((self.train_data, self.inference_data))
        mess_train_label = np.concatenate((self.train_label, mess_label))
        import time
        t1 = time.time()
        self.model.fit(mess_train_data, mess_train_label)
        t2 = time.time()
        log(f'train size:{len(mess_train_label)}')
        train_acc = (self.model.predict(mess_train_data) == mess_train_label).sum() / len(mess_train_label)
        log(f'train acc:{train_acc}')
        self.name = f'{self.name}-train{train_acc:.2f}@{(t2-t1)}s'

    def pca_preprocess(self, data, label, inference_data, n_dim, val_split=0.001):
        from preprocess import by_kernel_pca
        from preprocess import by_pca
        from preprocess import normlize
        from preprocess import row

        import numpy as np

        data = data.reshape((data.shape[0], -1))
        inference_data = inference_data.reshape((inference_data.shape[0], -1))
        all_data = np.concatenate((data, inference_data))
        pca_all_data, pca_ratio = row(all_data, n_dim)
        self.name = f'{self.name}-pcr{n_dim/64}@{pca_ratio:.2f}'
        # self.name = f'{self.name}-normlized'
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
        log(f'val svm acc: {acc:.4f} @[{hit}/{len(self.val_label)}]')
        self.name = f'{self.name}-vld{acc:.2f}'
        return acc

    def cinference(self, data):
        res = self.model.predict(self.inference_data)
        return res
