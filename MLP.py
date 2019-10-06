import numpy as np


class MultiLayerPerceptron(object):
    def __init__(self, hidden=30, L2=0, epochs=20, lr=.01, shuffle=True, mini_batch_size=1, seed=None):
        self.eval = {'cost': [],
                     'train_acc': [],
                     'valid_acc': []
                     }
        self.hidden = hidden
        self.L2 = L2
        self.epohcs = epochs
        self.lr = lr
        self.shuffe = shuffle
        self.mini_batch_size = mini_batch_size
        self.seed = seed

    def fit(self, X, y):
        n_output = np.unique(y).shape[0]  # 10 ouputs
        n_features = X.shape[1]  # 784 Features

        # Now we need to initalize weights for hidden and output layers

        # input -- > hidden
        self.bias_hidden = np.zeros(self.hidden)
        self.weights_hidden = np.random.rand(n_features, self.hidden)

        # hidden -- > output
        self.bias_output = np.zeros(n_output)
        self.weigths_output = np.random.rand(self.hidden, n_output)

        y_train_enc = self._onehot(y, n_output)


    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self):
        pass

    def _onehot(self, y, n_classes):

        one_hot = np.zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y.astype(int)):
            one_hot[val, idx] = 1
        return one_hot.T

    def compute_cost(self):
        pass

    def predict(self):
        pass
