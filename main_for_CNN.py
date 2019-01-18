import datahelper
import cnns
import argparse


def pad_3(data):
    import numpy as np
    return np.concatenate((data, data, data), axis=1)


def train():
    data, label = datahelper.get_dataset()
    # data = pad_3(data)
    train_data, train_label, val_data, val_label = datahelper.split_val_set(data, label, val_split=0.1, shuffle=True)

    cnn = cnns.SSCNN(name='val0.1-epoch20')
    cnn.ctrain(train_data, train_label, val_data, val_label, lr=1e-3, epoch_num=30)

    return cnn


def inference(model):
    data = datahelper.get_inference_data()
    data = pad_3(data)
    categories = model.cinference(data)
    datahelper.write_to_submission(list(categories), sname=f'{model.name}')
    print(f'saved: {model.name}')
    return


def load_then_inference(name='scnn-cc-0.123-0.036'):
    cnn = cnns.SSCNN()
    cnn.cload(name)
    inference(cnn)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='svm')
    parser.add_argument('--angle', type=int, default=90)
    parser.add_argument('--grid_size', type=int, default=4)
    return parser.parse_args()


if __name__ == '__main__':
    # arg = get_parser()
    model = train()
    # for i in range(1, 64, 1):
    #     print()
    #     print(i)
    #     model = get_SVM(pca_dim=64 * i)
    inference(model)
    # load_then_inference('scnn-cc-0.123-0.036')
