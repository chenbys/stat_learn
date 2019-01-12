import datahelper
import cnns
import vgg
import argparse


def pad_3(data):
    import numpy as np
    return np.concatenate((data, data, data), axis=1)


def train():
    data, label = datahelper.get_dataset()
    data = pad_3(data)
    class_num = len(set(label))
    train_data, train_label, val_data, val_label = datahelper.split_val_set(data, label, val_split=0.5, shuffle=True)

    # cnn = cnns.SSCNN(name='SSCNN-adam-val0.1-epoch50')
    # cnn.ctrain(train_data, train_label, val_data, val_label, lr=1e-3, epoch_num=50)
    # cnn.csave()

    cnn = vgg.VGG(name='val0.5-epoch50')
    cnn.load_pretrained()
    cnn.ctrain(train_data, train_label, val_data, val_label, lr=1e-3, epoch_num=50)

    # cnn.cload('val0.1-epoch50VGG11-27-0.089-0.057')
    # print('\n1e-3')
    # cnn.ctrain(train_data, train_label, val_data, val_label, lr=1e-3, epoch_num=20)
    # cnn.csave()
    #
    # print('\n1e-4')
    # cnn.ctrain(train_data, train_label, val_data, val_label, lr=1e-4, epoch_num=10)
    # cnn.csave()
    #
    # print('\n1e-5')
    # cnn.ctrain(train_data, train_label, val_data, val_label, lr=1e-5, epoch_num=10)
    # cnn.csave()
    #
    # print('\n1e-6')
    # cnn.ctrain(train_data, train_label, val_data, val_label, lr=1e-6, epoch_num=10)
    # cnn.csave()
    #
    # print('\n1e-7')
    # cnn.ctrain(train_data, train_label, val_data, val_label, lr=1e-7, epoch_num=10)
    # cnn.csave()

    # print(cnn.train_loss)
    # print(cnn.val_loss)
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


def get_SVM(pca_dim):
    data, label = datahelper.get_dataset()
    inference_data = datahelper.get_inference_data()

    from svms import SVM
    svm = SVM(data, label, inference_data, pca_dim=pca_dim)
    svm.ctrain()
    svm.cvalidate()
    return svm


def get_Adaboost():
    data, label = datahelper.get_dataset()
    inference_data = datahelper.get_inference_data()

    from ensemble import Adaboost
    model = Adaboost(data, label, inference_data)
    model.ctrain()
    model.cvalidate()
    return model


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
