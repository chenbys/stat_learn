# log_path = 'a.log'
# import matplotlib.pyplot as plt
#
# loss, acc = [], []
# with open(log_path) as f:
#     for line in f.readlines():
#         t = line.strip().split('loss: ')[1]
#         sloss, sacc = t.split(', acc: ')
#         loss.append(float(sloss))
#         acc.append(float(sacc))
#
# # plt.rcParams['figure.figsize'] = (20.0, 20.0)
# f = plt.figure()
# a1 = f.add_subplot(121)
# a2 = f.add_subplot(122)
#
# a1.plot(loss[0::2], linewidth=3, label='train')
# a1.plot(loss[1::2], linewidth=3, label='validate')
# a1.legend(loc='upper right', fontsize=10)
#
# a2.plot(acc[0::2], linewidth=3, label='train')
# a2.plot(acc[1::2], linewidth=3, label='validate')
# a2.legend(loc='upper left', fontsize=10)
#
# # plt.xticks(fontsize=30)
# # plt.yticks(fontsize=30)
# plt.show()
#
# a = 1


def plot_log(train_loss, train_acc, val_loss, val_acc):
    import matplotlib.pyplot as plt
    f = plt.figure()
    a1 = f.add_subplot(121)
    a2 = f.add_subplot(122)

    a1.plot(train_loss, linewidth=3, label='train')
    a1.plot(val_loss, linewidth=3, label='validate')
    a1.legend(loc='upper right', fontsize=10)

    a2.plot(train_acc, linewidth=3, label='train')
    a2.plot(val_acc, linewidth=3, label='validate')
    a2.legend(loc='upper left', fontsize=10)

    # plt.xticks(fontsize=30)
    # plt.yticks(fontsize=30)
    plt.show()
