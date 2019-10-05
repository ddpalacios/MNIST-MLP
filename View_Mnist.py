import numpy as np

mnist =  np.load('mnist_data.npz')

X_train, y_train, X_test, y_test = [mnist[f] for f in mnist.files]


print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)