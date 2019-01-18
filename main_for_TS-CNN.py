import datahelper
import cnns
import numpy as np


def train():
    val_split = 0.1
    data, label = datahelper.get_dataset()
    train_data, train_label, val_data, val_label = datahelper.split_val_set(data, label, val_split=val_split,
                                                                            shuffle=True)
    kaggle_data = datahelper.get_inference_data()
    kaggle_label = datahelper.get_mess_label()

    mess_train_data = np.concatenate((train_data, kaggle_data), axis=0)
    mess_train_label = np.concatenate((train_label, kaggle_label))

    cnn = cnns.SSCNN(name='SSCNN')
    train_loss, train_acc = [], []
    val_loss, val_acc = [], []

    for i in range(3):
        print('\nNormal training===================================================================')
        cnn.ctrain(train_data, train_label, val_data, val_label,
                   lr=1e-1, wd=1e-2, epoch_num=10)
        train_loss += cnn.train_loss
        train_acc += cnn.train_acc
        val_loss += cnn.val_loss
        val_acc += cnn.val_acc
        print('\nMess training========================================================================')
        cnn.ctrain(mess_train_data, mess_train_label, val_data, val_label,
                   lr=1e-1, wd=1e-2, epoch_num=10)
        train_loss += cnn.train_loss
        train_acc += cnn.train_acc
        val_loss += cnn.val_loss
        val_acc += cnn.val_acc
    return cnn


def inference(model):
    data = datahelper.get_inference_data()
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
