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


def get_dataset(type='train', ):
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


def get_inference_data(fname='test_new'):
    import os
    csv_file = 'data/' + fname + '.csv'
    npy_fname = 'data/' + fname + '.npy'
    if os.path.exists(npy_fname):
        data = np.load(npy_fname)
    else:
        ofile = csv.DictReader(open(csv_file))
        ids = []
        data = []
        for r in ofile:
            ids.append(r['id'])
            X = np.array([float(x) for x in list(r.values())[1:]])
            X = X.reshape((1, 64, 64))
            data.append(X)

        data = np.array(data)
        np.save(npy_fname, data)

    return data


def get_mess_label(fname='0.92200@svm1ovo-pcr768.00-train0.99-vld1.00.csv'):
    csv_file = f'data/{fname}'
    import pandas as pd
    a = pd.read_csv(csv_file)
    return a['categories'].tolist()


def write_to_submission(categories, sname='first'):
    import pandas as pd
    ids = range(len(categories))
    col_ids = pd.Series(ids, name='id')
    col_cts = pd.Series(categories, name='categories')
    con = pd.concat([col_ids, col_cts], axis=1)
    con.to_csv(f'submissions2/{sname}.csv', index=False, sep=',')


if __name__ == '__main__':
    A = get_mess_label()
