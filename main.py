# pylint: disable=pointless-statement, unnecessary-pass, unused-variable, unused-import
#!/usr/bin/env python3
# python main.py

import lib.data
import lib.helper
from PIL import Image
import numpy as np
import click
from lib.log import log

# from lib.helper import predict
from lib.data import load_dataset
from lib.plot import plot_image
from lib.plot import plot_scatter
from lib.linear_regression import model
from lib.linear_regression import predict


@click.group()
def cli():
    """run NN modelling and prediction"""
    pass


@cli.command()
@click.argument("example", type=int)
def injest(example):
    """Load a dataset ("data.h5") containing: - a training set of m_train images labeled as cat (y=1) or non-cat (y=0) - a test set of m_test images labeled as cat or non-cat - each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB). Thus, each image is square (height = num_px) and (width = num_px)"""

    # Loading the data (cat/non-cat)
    (
        train_set_x_orig,
        train_set_y_orig,
        test_set_x_orig,
        test_set_y_orig,
        train_set_x,
        test_set_x,
        train_set_y,
        test_set_y,
        classes,
    ) = load_dataset()
    # Decode the bytes to strings
    labels = [label.decode("utf-8") for label in classes]

    # visualize image from each line of train_set_x_orig / test_set_x_orig array

    a = "It is " + str(labels[int(train_set_y_orig[example,])])  # row_wise
    plot_image(train_set_x_orig[example], a)

    a = "It is " + str(labels[int(test_set_y_orig[example,])])  # row_wise
    plot_image(test_set_x_orig[example], a)

    click.echo(a)


@cli.command()
# @click.argument("digit", type=int)
def modeling():
    """Train Linear Regression model weights and bias to classify an image as Cat or not"""

    # Loading the data (cat/non-cat)
    (
        train_set_x_orig,
        train_set_y_orig,
        test_set_x_orig,
        test_set_y_orig,
        train_set_x,
        test_set_x,
        train_set_y,
        test_set_y,
        classes,
    ) = load_dataset()

    # train model
    logistic_regression_model = model(
        train_set_x,
        train_set_y,
        test_set_x,
        test_set_y,
        num_iterations=2000,
        learning_rate=0.005,
        print_cost=False,
    )

    # save model to an NPY file
    np.save("model/model_weights.npy", logistic_regression_model["w"])
    np.save("model/model_bias.npy", np.array([logistic_regression_model["b"]]))

    # save datasets to an NPY file
    # np.save("model/test_set_x.npy", test_set_x)
    # np.save("model/test_set_y.npy", test_set_y)

    # plot costs
    plot_scatter(logistic_regression_model["costs"])

    click.echo("Costs = " + str(np.squeeze(logistic_regression_model["costs"])))
    log("Costs = " + str(np.squeeze(logistic_regression_model["costs"])))

    click.echo(
        "Train_accuracy = "
        + str(np.squeeze(logistic_regression_model["train_accuracy"]))
    )
    click.echo(
        "Test_accuracy = " + str(np.squeeze(logistic_regression_model["test_accuracy"]))
    )
    log(
        "Train_accuracy = "
        + str(np.squeeze(logistic_regression_model["train_accuracy"]))
    )
    log(
        "Test_accuracy = " + str(np.squeeze(logistic_regression_model["test_accuracy"]))
    )


@cli.command()
@click.argument("example", type=int)
def predict_test(example):
    """Predict a example of test dataset referenced by index supplied in the argument"""

    # injest test datasets
    (
        train_set_x_orig,
        train_set_y_orig,
        test_set_x_orig,
        test_set_y_orig,
        train_set_x,
        test_set_x,
        train_set_y,
        test_set_y,
        classes,
    ) = load_dataset()

    # Decode the bytes to strings
    labels = [label.decode("utf-8") for label in classes]

    # Load trained model from the NPY file
    w = np.load("model/model_weights.npy")
    b = np.load("model/model_bias.npy")[
        0
    ]  # convert a Python array with a single element to a scalar

    # flatten and standardize
    test_set_x_example = (
        test_set_x_orig[example, :].reshape(1, -1).T / 255.0
    )  # 1 column, many rows of features
    a = "It is " + str(labels[int(test_set_y_orig[example,])])  # row_wise
    p = "Predicted as " + str(
        labels[int(np.squeeze(predict(w, b, test_set_x_example)))]
    )
    click.echo(a)
    click.echo(p)

    log("Example " + str(example) + " :: " + a + " :: " + p)

    # visualize image from each line of train_set_x_orig / test_set_x_orig array
    plot_image(test_set_x_orig[example], a)


@cli.command()
@click.argument("file_name")
def predict_unseen(file_name):
    """Predict unseen image supplied by file name in the argument"""

    # injest test datasets
    (
        train_set_x_orig,
        train_set_y_orig,
        test_set_x_orig,
        test_set_y_orig,
        train_set_x,
        test_set_x,
        train_set_y,
        test_set_y,
        classes,
    ) = load_dataset()
    # Decode the bytes to strings
    labels = [label.decode("utf-8") for label in classes]
    num_px = train_set_x_orig.shape[1]

    # injest image file my_image.jpg
    fname = "assets/images/" + file_name

    imageori = np.array(Image.open(fname))
    log("imageori shape: " + str(imageori.shape))
    click.echo("imageori shape " + str(imageori.shape))

    image = np.array(Image.open(fname).resize((num_px, num_px)))
    log("image.shape " + str(image.shape))
    click.echo("image.shape " + str(image.shape))

    image = image.reshape(1, -1).T / 255.0  # 1 column, many rows of features

    log("image.shape " + str(image.shape))
    click.echo("image.shape " + str(image.shape))

    # Load trained model from the NPY file
    w = np.load("model/model_weights.npy")
    b = np.load("model/model_bias.npy")[
        0
    ]  # convert a Python array with a single element to a scalar

    a = "Is it cat?"
    p = "predicted as " + str(labels[int(np.squeeze(predict(w, b, image)))])
    click.echo(p)

    log("Unseen image is " + p)

    # visualize image from each line of train_set_x_orig / test_set_x_orig array
    plot_image(imageori, a)


if __name__ == "__main__":
    cli()
