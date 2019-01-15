import datahelper
import cnns
import vgg
import numpy as np


def pad_3(data):
    return np.concatenate((data, data, data), axis=1)


def train():
    val_split = 0.1
    data, label = datahelper.get_dataset()
    data = pad_3(data)
    train_data, train_label, val_data, val_label = datahelper.split_val_set(data, label, val_split=val_split,
                                                                            shuffle=True)

    cnn = vgg.VGG('TS2-')
    cnn.ctrain(train_data, train_label, val_data, val_label,
               lr=5e-6, wd=1e0, epoch_num=50, val_split=val_split)

    kaggle_data = datahelper.get_inference_data()
    kaggle_data = pad_3(kaggle_data)
    kaggle_label = cnn.cinference(kaggle_data)

    mess_train_data = np.concatenate((train_data, kaggle_data), axis=0)
    mess_train_label = np.concatenate((train_label, kaggle_label))
    cnn.ctrain(mess_train_data, mess_train_label, val_data, val_label,
               lr=5e-6, wd=1e0, epoch_num=50, val_split=val_split)
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


if __name__ == '__main__':
    model = train()
    inference(model)
