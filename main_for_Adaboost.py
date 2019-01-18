import datahelper
import cnns
import vgg
import argparse


def inference(model):
    data = datahelper.get_inference_data()
    categories = model.cinference(data)
    datahelper.write_to_submission(list(categories), sname=f'{model.name}')
    print(f'saved: {model.name}')
    return


def get_Adaboost():
    data, label = datahelper.get_dataset()
    inference_data = datahelper.get_inference_data()

    from ensemble import Adaboost
    model = Adaboost(data, label, inference_data)
    model.ctrain()
    model.cvalidate()
    return model


if __name__ == '__main__':
    model = get_Adaboost()
    inference(model)
