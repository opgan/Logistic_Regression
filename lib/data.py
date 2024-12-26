# pylint: disable=E1101, unused-import

import numpy as np
import h5py
from lib.log import log


def load_dataset():
    train_dataset = h5py.File("datasets/train_catvnoncat.h5", "r")
    train_set_x_orig = np.array(
        train_dataset["train_set_x"][:]
    )  # your train set features
    train_set_y_orig = np.array(
        train_dataset["train_set_y"][:]
    )  # your train set labels

    test_dataset = h5py.File("datasets/test_catvnoncat.h5", "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    # Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)
    m_train = train_set_x_orig.shape[0]  # train_set_y.size
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]  # assume image is square

    # Reshape the training and test examples
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    # train_set_y_orig and test_set_y_orig are originally row vector (m, 1). Reshape to column vector (1,m)
    train_set_y = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    # standardize dataset
    train_set_x = train_set_x_flatten / 255.0
    test_set_x = test_set_x_flatten / 255.0

    print("Number of training examples: m_train = " + str(m_train))
    print("Number of testing examples: m_test = " + str(m_test))
    print("Height/Width of each image: num_px = " + str(num_px))
    print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ")")
    print("Original train_set_x shape: " + str(train_set_x_orig.shape))
    print("Original train_set_y shape: " + str(train_set_y_orig.shape))
    print("Original test_set_x shape: " + str(test_set_x_orig.shape))
    print("Original test_set_y shape: " + str(test_set_y_orig.shape))

    print("Flatten standardised train_set_x shape: " + str(train_set_x.shape))
    print("Flatten standardised test_set_x shape: " + str(test_set_x.shape))
    print("Row-wise train_set_y shape: " + str(train_set_y.shape))
    print("Row-wise test_set_y shape: " + str(test_set_y.shape))

    log("Number of training examples: m_train = " + str(m_train))
    log("Number of testing examples: m_test = " + str(m_test))
    log("Height/Width of each image: num_px = " + str(num_px))
    log("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ")")
    log("Original train_set_x shape: " + str(train_set_x_orig.shape))
    log("Original train_set_y shape: " + str(train_set_y_orig.shape))
    log("Original test_set_x shape: " + str(test_set_x_orig.shape))
    log("Original test_set_y shape: " + str(test_set_y_orig.shape))
    log("Flatten standardised train_set_x shape: " + str(train_set_x.shape))
    log("Flatten standardised test_set_x shape: " + str(test_set_x.shape))
    log("Row-wise train_set_y shape: " + str(train_set_y.shape))
    log("Row-wise  test_set_y shape: " + str(test_set_y.shape))

    return (
        train_set_x_orig,
        train_set_y_orig,
        test_set_x_orig,
        test_set_y_orig,
        train_set_x,
        test_set_x,
        train_set_y,
        test_set_y,
        classes,
    )
