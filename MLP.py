import sys

import numpy as np
from time import sleep


class MultiLayerPerceptron(object):
    def __init__(self, hidden_units=100, L2=0., epochs=20, lr=.01, shuffle=True, mini_batch_size=1, seed=None):
        self.eval = {'cost': [],
                     'train_acc': [],
                     'valid_acc': []
                     }
        self.random = np.random.RandomState(seed)
        self.hidden_units = hidden_units
        self.L2 = L2
        self.epochs = epochs
        self.lr = lr
        self.shuffle = shuffle
        self.mini_batch_size = mini_batch_size
        self.seed = seed

    def fit(self, X, y):
        # Set up
        n_output = np.unique(y).shape[0]  # 10 outputs
        n_features = X.shape[1]  # 784 Features

        # Initialize weights to form MLP model
        # input -- > hidden
        self.weights_hidden = self.random.normal(loc=0.0, scale=0.1,
                                                 size=(n_features, self.hidden_units))  # (784 , 100)
        self.bias_hidden = np.zeros(self.hidden_units)  # (100, )

        # hidden -- > output
        self.weigths_output = self.random.normal(loc=0.0, scale=0.1, size=(self.hidden_units, n_output))  # (100, 10)
        self.bias_output = np.zeros(n_output)  # (10, )

        # Lets print out what our model may look like
        self.print_model_architecture()
        print("Will start activating weights and bias...\n\n")
        sleep(2)
        epoch_strlen = len(str(self.epochs))  # for progr format
        y_train_enc = self.onehot(y, n_output)  # Q.  Why is this method important? How does this affect our data?

        for epochs in range(self.epochs):
            indices = np.arange(X.shape[0])

            if self.shuffle:  # Shuffles indices
                self.random.shuffle(indices)

            for start_idx in range(0, indices.shape[0], self.mini_batch_size):
                batch_idx = indices[start_idx:start_idx + self.mini_batch_size]  # batch idx = [1: 100]

                net1, out1, net2, out2 = self.forward(X[batch_idx])

                ################
                # Backpropagation
                #################
                self.backward(X, out1, out2, y_train_enc, batch_idx)

            ###########
            # Evaluation
            ###########
            self.evaluation(X, y, y_train_enc, epoch_strlen, epochs)

        return self

    def forward(self, X):
        # FEED FORWARD DATA

        # ACTIVATE Input --> hidden
        net1 = np.dot(X, self.weights_hidden) + self.bias_hidden

        out1 = self.sigmoid(net1)

        # ACTIVATE hidden --> output
        net2 = np.dot(out1, self.weigths_output) + self.bias_output
        out2 = self.sigmoid(net2)

        # self.print_live_forward_activations(out1, out2)

        return net1, out1, net2, out2

    def backward(self, X, out1, out2, y_train_enc, batch_idx):

        sigma_out = out2 - y_train_enc[batch_idx]
        sig_derivative = out1 * (1. - out1)
        sigma_h = (np.dot(sigma_out, self.weigths_output.T) * sig_derivative) # Reverse steps



        grad_w_h = np.dot(X[batch_idx].T, sigma_h)
        grad_w_out = np.dot(out1.T, sigma_out)


        grad_b_h = np.sum(sigma_h, axis=0)
        grad_b_out = np.sum(sigma_out, axis=0)



        delta_w_h = (grad_w_h + self.L2 * self.weights_hidden) # Regularization implementation
        delta_w_out = (grad_w_out + self.L2 * self.weigths_output)  # Regularization implementation

        delta_b_h = grad_b_h
        delta_b_out = grad_b_out


        self.weights_hidden -= self.lr * delta_w_h
        self.bias_hidden -= self.lr * delta_b_h
        self.weigths_output -= self.lr * delta_w_out
        self.bias_output -= self.lr * delta_b_out




    def evaluation(self, X, y, y_train_enc, epoch_strlen, epochs):
        net1, out1, net2, out2 = self.forward(X)
        cost = self.compute_cost(y_enc=y_train_enc, output=out2)

        y_train_pred = self.predict(X)
        train_acc = ((np.sum(y == y_train_pred)).astype(np.float) /
                     X.shape[0])

        sys.stderr.write('\r%0*d/%d| Cost: %.2f '
                         '| Train.: %.2f%%' % (epoch_strlen, epochs + 1, self.epochs, cost, train_acc * 100))

        sys.stderr.flush()

        self.eval['cost'].append(cost)
        self.eval['train_acc'].append(train_acc)

    def sigmoid(self, z):
        try:

            return 1 / (1 + np.exp(-z))

        except:
            pass

    def onehot(self, labels, n_classes):
        one_hot = np.zeros((n_classes, labels.shape[0]))  # (10, 6000)

        for idx, elements in enumerate(labels.astype(int)):
            # print("Storing 1 in {} Row and {} column\n\n".format(elements+1, idx+1))
            one_hot[elements, idx] = 1
        return one_hot.T

    def compute_cost(self, y_enc, output):
        L2_term = (self.L2 *
                   (np.sum(self.weights_hidden ** 2.) +
                    np.sum(self.weigths_output ** 2.)))

        term1 = -y_enc * (np.log(output))
        term2 = (1. - y_enc) * np.log(1. - output)
        cost = np.sum(term1 - term2) + L2_term
        return cost

    def predict(self, X):
        net1, out1, net2, out2 = self.forward(X)

        y_pred = np.argmax(out2, axis=1)


        return y_pred

    def print_model_architecture(self):
        print(
            "Model Architecture:\n\nHidden weights: {}\nHidden Bias: {}\nOutput weights: {}\nOutput Bias: {}".format(
                self.weights_hidden.shape,
                self.bias_hidden.shape,
                self.weigths_output.shape,
                self.bias_output.shape))

    def print_live_forward_activations(self, out1, out2):
        print("Activation from Input -> Hidden\n{}\n\nActivation from Hidden -> Output\n{}\n\n".format(out1, out2))
