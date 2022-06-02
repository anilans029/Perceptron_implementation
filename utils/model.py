import numpy as np
import os
import joblib
import logging as log


class Perceptron:
    def __init__(self, eta: float = None, epochs: int = None):
        self.weights = np.random.randn(3) * 1e-4  ### small random weights
        training = (eta is not None) and (epochs is not None)
        if training:
            log.info(f"initial weights before training :  \n{self.weights}")
        self.eta = eta
        self.epochs = epochs

    def _z_outcome(self, inputs, weights):
        return np.dot(inputs, weights)

    def activation_function(self, z):
        return np.where(z > 0, 1, 0)

    def fit(self, X, y):
        self.X = X
        self.y = y

        X_with_bias = np.c_[self.X, -np.ones((len(self.X), 1))]
        log.info(f"X with bias :\n {X_with_bias}")

        for epoch in range(self.epochs):
            log.info("--" * 13)
            log.info(f"for epoch >> {epoch + 1}")
            log.info("--" * 13)

            z = self._z_outcome(X_with_bias, self.weights)
            y_hat = self.activation_function(z)
            log.info(f"predicted value after forward pass : \n{y_hat}")

            self.error = self.y - y_hat
            log.info(f"error : \n{self.error}")

            self.weights = self.weights + self.eta * np.dot(X_with_bias.T, self.error)
            log.info(f"updated weights after epoch : {epoch + 1}/{self.epochs}: \n {self.weights}")
            log.info("##" * 13)

    def prediction(self, test_inputs):
        X_with_bias = np.c_[test_inputs, -np.ones((len(test_inputs), 1))]
        z = self._z_outcome(X_with_bias, self.weights)
        return self.activation_function(z)

    def total_loss(self):
        total_loss = np.sum(self.error)
        log.info(f"total loss : {total_loss}\n")

    def _create_dir_return_path(self, model_dir, filename):
        os.makedirs(model_dir, exist_ok=True)
        return os.path.join(model_dir, filename)

    def save(self, filenamem, model_dir=None):
        if model_dir is not None:
            model_file_path = self._create_dir_return_path(model_dir, filenamem)
            joblib.dump(self, model_file_path)

        else:
            model_file_path = self._create_dir_return_path("model", filenamem)
            joblib.dump(self, model_file_path)

    def load(self, filepath):
        return joblib.load(filepath)
