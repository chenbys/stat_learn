import datahelper
import solutions


def main():
    data, label = datahelper.get_dataset()
    class_num = len(set(label))
    train_data, train_label, val_data, val_label = datahelper.split_val_set(data, label, val_split=0.2, shuffle=True)
    cnn = solutions.CNN(class_num)
    cnn.ctrain(train_data, train_label, val_data, val_label)
    return


if __name__ == '__main__':
    main()
