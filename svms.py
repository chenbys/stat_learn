from sklearn import svm


class SVM(object):
    def __init__(self, data, label, inference_data, name='svm'):
        self.name = name
        self.preprocess(data, label, inference_data)
        self.clf = svm.SVC()

    def ctrain(self):
        self.clf.fit(self.train_data, self.train_label)

    def preprocess(self, data, label, inference_data, n_dim=64 * 6, val_split=0.2):
        from reduce_dimensions import by_kernel_pca
        from reduce_dimensions import by_pca

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
        res = self.clf.predict(self.val_data)
        hit = (res == self.val_label).sum()
        acc = hit / len(self.val_label)
        print(f'val svm acc: {acc:.4f} @[{hit}/{len(self.val_label)}]')
        self.name = f'acc{val:.2f}-{self.name}'
        return acc

    def cinference(self, data):
        res = self.clf.predict(self.inference_data)
        return res
