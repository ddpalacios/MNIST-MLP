import numpy as np


class MultiLayerPerceptron(object):
    def __init__(self, hidden_units=100, L2=0, epochs=20, lr=.01, shuffle=True, mini_batch_size=1, seed=None):
        self.hidden_units = hidden_units
        self.L2 = L2
        self.epohcs = epochs
        self.lr = lr
        self.shuffe = shuffle
        self.mini_batch_size = mini_batch_size
        self.seed = seed

    def fit(self, X, y):
        n_output = np.unique(y).shape[0]  # 10 ouputs
        n_features = X.shape[1]  # 784 Features

        # Now we need to initialize weights for hidden and output layers

        # input -- > hidden
        self.weights_hidden = np.random.rand(n_features, self.hidden_units)  # (784 , 100)
        self.bias_hidden = np.zeros(self.hidden_units)  # (100, )

        # hidden -- > output
        self.weigths_output = np.random.rand(self.hidden_units, n_output)  # (100, 784)
        self.bias_output = np.zeros(n_output)  # (10, )

        self.print_model_architecture()

        # y_train_enc = self.onehot(y, n_output)

        self.eval = {'cost': [],
                     'train_acc': [],
                     'valid_acc': []
                     }

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self):
        pass

    def onehot(self, y, n_classes):
        one_hot = np.zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y.astype(int)):
            one_hot[val, idx] = 1
        return one_hot.T

    def compute_cost(self):
        pass

    def predict(self):
        pass

    def print_model_architecture(self):
        print(
            "Model Architecture:\n\nHidden weights: {}\nHidden Bias: {}\nOutput weights: {}\nOutput Bias: {}".format(self.weights_hidden.shape,
                                                                                              self.bias_hidden.shape,
                                                                                              self.weigths_output.shape,
                                                                                              self.bias_output.shape))
