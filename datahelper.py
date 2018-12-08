import csv
import numpy as np
import matplotlib.pyplot as plt


def load_dataset_from_csv(set_file='data/train.csv'):
    file = csv.DictReader(open(set_file))
    ids = []
    data = []
    label = []
    for r in file:
        ids.append(r['id'])
        label.append(int(r['categories']))
        X = np.array([float(x) for x in list(r.values())[1:-1]])
        X = X.reshape((1, 64, 64))
        data.append(X)

    data, label = np.array(data), np.array(label)
    return data, label


def get_dataset(type='train'):
    import os
    file = 'data/' + type + '.csv'
    data_fname, label_fname = 'data/' + type + '_data.npy', 'data/' + type + '_label.npy'
    if os.path.exists(data_fname):
        data, label = np.load(data_fname), np.load(label_fname)
    else:
        data, label = load_dataset_from_csv(file)
        np.save(data_fname, data)
        np.save(label_fname, label)

    return data, label


def split_val_set(data, label, val_split=0.2, shuffle=False):
    if shuffle:
        idx = np.random.permutation(len(label))
        data = data[idx]
        label = label[idx]
    val_idx = int(val_split * len(label))
    return data[val_idx:], label[val_idx:], data[:val_idx], label[:val_idx]


if __name__ == '__main__':
    A, B = get_dataset()
