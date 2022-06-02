import pandas as pd
from utils.model import Perceptron
from utils.all_utils import prepare_data, save_plot

def main(data, modelname, plotname, eta, epochs):
     df_and = pd.DataFrame(data)
     X, y = prepare_data(df_and)

     model_and = Perceptron(eta=eta, epochs=epochs)
     model_and.fit(X, y)
     _ = model_and.total_loss()

     model_and.save(modelname)
     save_plot(df_and, model_and, fielname=plotname)
if __name__ == "__main__":
    AND = {
        "X1": [0,0,1,1],
        "X2": [0,1,0,1],
        "y" : [0,0,0,1]
    }
    ETA = 0.1
    EPOCHs = 10
    main(data = AND, modelname = "and.model", plotname = "and.png", eta = ETA, epochs= EPOCHs)



