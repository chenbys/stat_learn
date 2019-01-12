from sklearn import svm


class SVM(object):
    def __init__(self, data, label, inference_data, name='SVM', pca_dim=64 * 20):
        self.model = svm.SVC(gamma='scale', kernel='poly', degree=1, decision_function_shape='ovo')
        self.name = f'{name}ScalePoly1OVOval0.99'
        self.pca_preprocess(data, label, inference_data, n_dim=pca_dim)

    def ctrain(self):
        import time
        t1 = time.time()
        self.model.fit(self.train_data, self.train_label)
        t2 = time.time()
        print(f'train size:{len(self.train_label)}')
        train_acc = (self.model.predict(self.train_data) == self.train_label).sum() / len(self.train_label)
        print(f'train acc:{train_acc}')
        self.name = f'{self.name}-train{train_acc:.2f}@{(t2-t1)}s'

    def pca_preprocess(self, data, label, inference_data, n_dim, val_split=0.99):
        from preprocess import by_kernel_pca
        from preprocess import by_pca
        from preprocess import normlize
        from preprocess import row

        import numpy as np

        data = data.reshape((data.shape[0], -1))
        inference_data = inference_data.reshape((inference_data.shape[0], -1))
        all_data = np.concatenate((data, inference_data))
        pca_all_data, pca_ratio = by_pca(all_data, n_dim)
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
        print(f'val svm acc: {acc:.4f} @[{hit}/{len(self.val_label)}]')
        self.name = f'{self.name}-vld{acc:.2f}'
        return acc

    def cinference(self, data):
        res = self.model.predict(self.inference_data)
        return res
