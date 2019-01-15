import datahelper


def inference(model):
    data = datahelper.get_inference_data()

    categories = model.cinference(data)
    datahelper.write_to_submission(list(categories), sname=f'{model.name}')
    print(f'saved: {model.name}')
    return


def get_SVM(pca_dim=64 * 64):
    data, label = datahelper.get_dataset()
    inference_data = datahelper.get_inference_data()
    mess_label = datahelper.get_mess_label()

    from svms import SVM
    svm = SVM(data, label, inference_data, pca_dim=pca_dim, name='TS-SVM')
    for i in range(5):
        svm.ctrain()
        svm.cvalidate()

        svm.TS_train(mess_label)
        svm.cvalidate()

    svm.ctrain()
    svm.cvalidate()
    print('inference')
    inference(svm)
    return svm


if __name__ == '__main__':
    model = get_SVM()
