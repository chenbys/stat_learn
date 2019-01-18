import datahelper
import argparse


def inference(model):
    data = datahelper.get_inference_data()
    categories = model.cinference(data)
    datahelper.write_to_submission(list(categories), sname=f'{model.name}')
    print(f'saved: {model.name}')
    return


def get_SVM(pca_dim):
    data, label = datahelper.get_dataset()
    inference_data = datahelper.get_inference_data()

    from svms import SVM
    svm = SVM(data, label, inference_data, pca_dim=pca_dim)
    svm.ctrain()
    svm.cvalidate()
    return svm


if __name__ == '__main__':
    model = get_SVM(pca_dim=64 * 64)
    inference(model)
