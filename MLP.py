import numpy as np


class MultiLayerPerceptron(object):
    def __init__(self, hidden_units=100, L2=0, epochs=20, lr=.01, shuffle=True, mini_batch_size=1, seed=None):
        self.random = np.random.RandomState(seed)
        self.hidden_units = hidden_units
        self.L2 = L2
        self.epochs = epochs
        self.lr = lr
        self.shuffle = shuffle
        self.mini_batch_size = mini_batch_size
        self.seed = seed

    def fit(self, X, y):
        n_output = np.unique(y).shape[0]  # 10 outputs
        n_features = X.shape[1]  # 784 Features

        # Initialize weights to form MLP model
        # input -- > hidden
        self.weights_hidden = np.random.rand(n_features, self.hidden_units)  # (784 , 100)
        self.bias_hidden = np.zeros(self.hidden_units)  # (100, )

        # hidden -- > output
        self.weigths_output = np.random.rand(self.hidden_units, n_output)  # (100, 784)
        self.bias_output = np.zeros(n_output)  # (10, )

        # Lets print out what our model may look like
        self.print_model_architecture()

        y_train_enc = self.onehot(y, n_output)  # Q.  Why is this method important? How does this affect our data?

        self.eval = {'cost': [],
                     'train_acc': [],
                     'valid_acc': []
                     }

        for _ in range(self.epochs):
            indices = np.arange(X.shape[0])

            if self.shuffle:  # Shuffles indices
                self.random.shuffle(indices)

            for start_idx in range(0, indices.shape[0] - \
                                      self.mini_batch_size + \
                                      1, self.mini_batch_size):
                batch_idx = indices[start_idx:start_idx + \
                                              self.mini_batch_size]

                print(batch_idx)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self):
        pass

    def onehot(self, labels, n_classes):
        one_hot = np.zeros((n_classes, labels.shape[0]))  # (10, 6000)

        for idx, elements in enumerate(labels.astype(int)):
            # print("Storing 1 in {} Row and {} column\n\n".format(elements+1, idx+1))
            one_hot[elements, idx] = 1
        return one_hot.T

    def compute_cost(self):
        pass

    def predict(self):
        pass

    def print_model_architecture(self):
        print(
            "Model Architecture:\n\nHidden weights: {}\nHidden Bias: {}\nOutput weights: {}\nOutput Bias: {}".format(
                self.weights_hidden.shape,
                self.bias_hidden.shape,
                self.weigths_output.shape,
                self.bias_output.shape))
