from MLP import MultiLayerPerceptron
from fileLoader import load_file
import numpy as np

df = load_file('mnist_data.npz')
X_train, y_train, X_test, y_test = df.get_data()

mlp = MultiLayerPerceptron(hidden_units=100, L2=.01, epochs=200, lr=.0005, shuffle=True, mini_batch_size=100, seed=1)


mlp.fit(X_train[:55000], y_train[:55000])



