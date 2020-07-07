
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bhtsne import tsne


def get_random_block_from_data_and_label(data, label, size):
    sample = np.random.randint(len(data), size=size)
    return data[sample], label[sample]


def get_random_sample_from_data(data, size):
    sample = np.random.randint(len(data), size=size)
    return sample


def plot_tsne_for_different_motor_left(x, y, ax, motor):
    ax.scatter(x[:, 0], x[:, 1], c=y, s=10)
    # ax.legend(loc="upper right")
    ax.set_title("t-sne for lstm feature of motor {}".format(motor))

    colors = [plt.cm.jet(float(i) / 10) for i in range(10)]
    for i in range(10):
        colors[i] = np.array(colors[i])
        colors[i] = np.reshape(colors[i], (1, len(colors[i])))

        ax.scatter(
            x[y == i][:, 0],
            x[y == i][:, 1],
            c=colors[i],
            s=30,
            label='class_' + str(i))

def plot_tsne_for_different_motor_right(x, y, ax, motor):
    ax.scatter(x[:, 0], x[:, 1], c=y, s=10)
    ax.legend(loc="upper right")
    ax.set_title("t-sne for lstm feature of motor {}".format(motor))

    colors = [plt.cm.jet(float(i) / 10) for i in range(10)]
    for i in range(10):
        colors[i] = np.array(colors[i])
        colors[i] = np.reshape(colors[i], (1, len(colors[i])))

        ax.scatter(
            x[y == i][:, 0],
            x[y == i][:, 1],
            c=colors[i],
            s=30,
            label='class_' + str(i))
    ax.legend(loc='upper left', bbox_to_anchor=(1., 0.7))


# train_motor = 1
# test_motor = 3

# path = '../../base_model/saved_model/2019_07_18_16_46_38_cnn_lstm_sliding_20_motor_train_1_test_3'

def plot(ifPlot=False, path, train_motor, test_motor):
    train_lstm_feature = np.load(
        '{}/train_lstm_feature.npy'.format(path))
    train_lstm_feature = np.reshape(train_lstm_feature, (train_lstm_feature.shape[0], -1))
    test_lstm_feature = np.load(
        '{}/test_lstm_feature.npy'.format(path))
    test_lstm_feature = np.reshape(test_lstm_feature, (test_lstm_feature.shape[0], -1))

    train_labels = np.load(
        '{}/train_label.npy'.format(path))
    test_labels = np.load(
        '{}/test_label.npy'.format(path))
    # train_labels = np.reshape(train_labels, (train_labels.shape[0], 1))
    # test_labels = np.reshape(test_labels, (test_labels.shape[0], 1))

    train_lstm_feature = pd.DataFrame(train_lstm_feature)
    train_lstm_feature = train_lstm_feature.fillna(0)
    train_lstm_feature = train_lstm_feature.astype('float64')


    test_lstm_feature = pd.DataFrame(test_lstm_feature)
    test_lstm_feature = test_lstm_feature.fillna(0)
    test_lstm_feature = test_lstm_feature.astype('float64')


    sample_size = 2000
    train_sample = random.sample(range(0, train_lstm_feature.shape[0]), sample_size)
    test_sample = random.sample(range(0, test_lstm_feature.shape[0]), sample_size)

    # train_tsne_result = tsne(train_lstm_feature.iloc[train_sample])

    fit_feature = pd.concat((train_lstm_feature.iloc[train_sample], test_lstm_feature.iloc[test_sample]), axis=0)
    tsne_result = tsne(fit_feature)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    np.save('{}/lstm_feature_tsne.npy'.format(path), tsne_result[:sample_size])
    plot_tsne_for_different_motor_left(tsne_result[:sample_size], np.reshape(train_labels[train_sample], (sample_size)), axes[0], train_motor)


    # test_tsne_result = tsne(test_lstm_feature.iloc[test_sample])
    # np.save('{}/test_lstm_feature_tsne.npy'.format(path), tsne_result[sample_size:])
    plot_tsne_for_different_motor_right(tsne_result[sample_size:], np.reshape(test_labels[test_sample], (sample_size)), axes[1], test_motor)
    plt.savefig('{}/lstm_feature_tsne.jpg'.format(path))
    if ifPlot:
        plt.show()

