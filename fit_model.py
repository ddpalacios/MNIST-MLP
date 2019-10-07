from MLP import MultiLayerPerceptron
from fileLoader import load_file
import numpy as np

df = load_file('mnist_data.npz')
X_train, y_train, X_test, y_test = df.get_data()

mlp = MultiLayerPerceptron(hidden_units=100, L2=0, epochs=200, lr=.001, shuffle=True, mini_batch_size=1, seed=1)


mlp.fit(X_train, y_train)



