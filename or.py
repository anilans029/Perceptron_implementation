import pandas as pd
from utils.model import Perceptron
from utils.all_utils import prepare_data, save_plot
import logging as log
import os


log_dir = "logs"
gate = "or gate"
os.makedirs(log_dir,exist_ok=True)
log.basicConfig(filemode="a",
                filename=os.path.join(log_dir,"running_log.log"),
                level = log.INFO,
                format = "[%(levelname)s --%(asctime)s --%(module)s] ===>  %(message)s"
                )


def main(data, modelname, plotname, eta, epochs):
    """
    This function takes in a data set, a model name, a plot name, a learning rate, and the number of epochs to train for,
    and then trains a model on the data set, plots the loss, and saves the model.

    :param data: the data file
    :param modelname: the name of the model you want to train
    :param plotname: the name of the plot that will be saved
    :param eta: learning rate
    :param epochs: number of epochs to train the model
    """
    df = pd.DataFrame(data)
    log.info(f"the rawdata is \n{df}")
    X, y = prepare_data(df)

    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)
    _ = model.total_loss()

    model.save(modelname)
    save_plot(df, model, fielname=plotname)
if __name__ == "__main__":
    OR = {
        "X1": [0,0,1,1],
        "X2": [0,1,0,1],
        "y" : [0,1,1,1]
    }
    ETA = 0.1
    EPOCHs = 10
    try:
        log.info(f"<<<<< starting of the training {gate} >>>>>>")
        main(data = OR,modelname = "or.model", plotname = "or.png", eta = ETA, epochs= EPOCHs)
        log.info(f">>>>>> ending of the training {gate} <<<<<<<< \n\n\n")
    except Exception as e:
        log.exception(e)
        raise e