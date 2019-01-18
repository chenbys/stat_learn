from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import time


class Adaboost(object):
    def __init__(self, data, label, inference_data, learning_rate=1., n_estimators=200, max_depth=4, name='adaboost'):
        self.name = f'{n_estimators}base{max_depth}-{name}'
        self.preprocess(data, label, inference_data)
        dt_stump = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=1)
        dt_stump.fit(self.train_data, self.train_label)
        dt_stump_acc = dt_stump.score(self.train_data, self.train_label)
        print(f'stump acc: {dt_stump_acc}')

        self.model = AdaBoostClassifier(
            base_estimator=dt_stump,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            algorithm="SAMME.R")

    def ctrain(self):
        t1 = time.time()
        self.model.fit(self.train_data, self.train_label)
        print(f'time for train: {(time.time()-t1)/60:.1f}m')
        train_acc = (self.model.predict(self.train_data) == self.train_label).sum() / len(self.train_label)
        print(f'train acc:{train_acc}')
        self.name = f'train{train_acc:.2f}-{self.name}'
        return

    def preprocess(self, data, label, inference_data, n_dim=64 * 16, val_split=0.2):
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
