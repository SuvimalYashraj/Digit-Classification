import pickle
import gzip

import numpy as np

def load_data(train_image_csv, train_label_csv, test_image_csv):

    train_label = np.genfromtxt(train_label_csv, dtype=int, delimiter=',')
    train_label_one_hot = [one_hot_encoding(tr) for tr in train_label]
    train_image = np.genfromtxt(train_image_csv, dtype=int, delimiter=',')
    training_data = [train_image, np.asarray(train_label_one_hot)]

    test_image = np.genfromtxt(test_image_csv, dtype=int, delimiter=',')

    # with open("train_image.csv", "r") as ti:
    #     train_image = np.genfromtxt(ti, dtype=int, delimiter=',')
    
    # with open("train_label.csv", "r") as tl:
    #     train_label = np.genfromtxt(tl, dtype=int, delimiter=',')
    # with open("test_image.csv", "r") as tei:
    #     test_image = np.genfromtxt(tei, dtype=int, delimiter=',')

    # training_data = [train_image[:50000], np.asarray(train_label_one_hot[:50000])]
    # validation_data = [train_image[50000:], np.asarray(train_label_one_hot[50000:])]
    # test_label = np.genfromtxt('test_label.csv', dtype=int, delimiter=',')
    # test_label_one_hot = [vectorized_result(te) for te in test_label]
    # test_data = [test_image, np.asarray(test_label_one_hot)]

    return (training_data, test_image)

def one_hot_encoding(j):
    e = np.zeros(10, dtype=int)
    e[j] = 1
    return e