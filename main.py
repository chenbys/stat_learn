import datahelper
import cnns


def train():
    data, label = datahelper.get_dataset()
    class_num = len(set(label))
    train_data, train_label, val_data, val_label = datahelper.split_val_set(data, label, val_split=0.2, shuffle=True)
    cnn = cnns.SCNN(class_num)
    # cnn.cload('2nd-cc-0.0000')

    print('\n1e-3')
    cnn.ctrain(train_data, train_label, val_data, val_label, lr=1e-3, epoch_num=20)
    cnn.csave()

    print('\n1e-4')
    cnn.ctrain(train_data, train_label, val_data, val_label, lr=1e-4, epoch_num=10)
    cnn.csave()

    print('\n1e-5')
    cnn.ctrain(train_data, train_label, val_data, val_label, lr=1e-5, epoch_num=10)
    cnn.csave()

    print('\n1e-6')
    cnn.ctrain(train_data, train_label, val_data, val_label, lr=1e-6, epoch_num=10)
    cnn.csave()

    print('\n1e-7')
    cnn.ctrain(train_data, train_label, val_data, val_label, lr=1e-7, epoch_num=10)
    cnn.csave()

    print(cnn.train_loss)
    print(cnn.val_loss)
    return cnn


def inference(model):
    data = datahelper.get_inference_data()
    categories = model.cinference(data)
    datahelper.write_to_submission(categories.tolist(), sname=f'{model.name}-eighth')
    print('saved')
    return


def load_then_inference(name='scnn-cc-0.123-0.036'):
    cnn = cnns.SSCNN()
    cnn.cload(name)
    inference(cnn)


def get_SVM():
    data, label = datahelper.get_dataset()
    inference_data = datahelper.get_inference_data()

    from svms import SVM
    svm = SVM(data, label, inference_data)
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


if __name__ == '__main__':
    # model = train()
    model = get_Adaboost()
    inference(model)
    # load_then_inference('scnn-cc-0.123-0.036')
