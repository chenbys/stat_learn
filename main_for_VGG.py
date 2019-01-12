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
    val_split = 0.001
    train_data, train_label, val_data, val_label = datahelper.split_val_set(data, label,
                                                                            val_split=val_split, shuffle=True)
    cnn = vgg.VGG('dropout_smlrbgwd')
    # cnn.load_pretrained()
    cnn.ctrain(train_data, train_label, val_data, val_label, lr=5e-6, wd=1e0, epoch_num=50, val_split=val_split)
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
    model = train()
    inference(model)
