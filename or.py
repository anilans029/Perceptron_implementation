import pandas as pd
from utils.model import Perceptron
from utils.all_utils import prepare_data, save_plot

def main(data, modelname, plotname, eta, epochs):
     df_OR = pd.DataFrame(data)
     X, y = prepare_data(df_OR)

     model_or = Perceptron(eta=eta, epochs=epochs)
     model_or.fit(X, y)
     _ = model_or.total_loss()

     model_or.save(modelname)
     save_plot(df_OR, model_or, fielname=plotname)
if __name__ == "__main__":
    OR = {
        "X1": [0,0,1,1],
        "X2": [0,1,0,1],
        "y" : [0,1,1,1]
    }
    ETA = 0.1
    EPOCHs = 10
    main(data = OR,modelname = "or.model", plotname = "or.png", eta = ETA, epochs= EPOCHs)



