import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import ListedColormap
import logging as log


def prepare_data(df, target_col="y"):
    """
    It takes a dataframe and a target column name, and returns a tuple of the dataframe with the target column removed, and
    the target column.

    :param df: the dataframe you want to prepare
    :param target_col: the name of the column you want to predict, defaults to y (optional)
    """
    log.info("preparing the data for training")
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    return X, y


def save_plot(df, model, fielname="plot.png", plot_dir="plots"):
    """
    > This function takes a dataframe, a model, and a filename, and saves a plot of the model's predictions on the dataframe
    to the specified filename in the specified directory

    :param df: the dataframe that contains the data to be plotted
    :param model: the model object
    :param fielname: the name of the file to save the plot as, defaults to plot.png (optional)
    :param plot_dir: The directory where the plot will be saved, defaults to plots (optional)
    """
    def _create_base_plot(df):
        """
        > This function takes a dataframe and returns a base plot with the dataframe's columns as the x and y axes

        :param df: The dataframe to be plotted
        """
        log.info("creating the base plot for the data")
        df.plot(kind="scatter", x="X1", y="X2", c="y", s=100, cmap="coolwarm")
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
        plt.axvline(x=0, color="black", linestyle="--", linewidth=1)

        figure = plt.gcf()
        figure.set_size_inches(10, 8)

    def _plot_decision_region(X, y, classifier, resolution=0.02):
        """
        It plots the decision boundary of a classifier

        :param X: the feature matrix
        :param y: the target labels
        :param classifier: the classifier object
        :param resolution: the step size of the mesh grid
        """
        log.info("plotting the decision regions")
        colors = ("cyan", "lightgreen")
        cmap = ListedColormap(colors)

        X = X.values
        x1 = X[:, 0]
        x2 = X[:, 1]

        x1_min, x1_max = x1.min() - 1, x1.max() + 1
        x2_min, x2_max = x2.min() - 1, x2.max() + 1

        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))
        y_hat = classifier.prediction(np.array([xx1.ravel(), xx2.ravel()]).T)
        y_hat = y_hat.reshape(xx1.shape)

        plt.contourf(xx1, xx2, y_hat, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        plt.plot()

    X, y = prepare_data(df)
    _create_base_plot(df)
    _plot_decision_region(X, y, model)

    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, fielname)
    log.info(f"saving the plot at path {plot_path}")
    plt.savefig(plot_path)