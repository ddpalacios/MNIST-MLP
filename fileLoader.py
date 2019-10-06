import numpy as np
import matplotlib.pyplot as plt
class load_file:
    def __init__(self, file):
        self.file = file


    def get_data(self):
        mnist = np.load(self.file)
        X_train, y_train, X_test, y_test = [mnist[f] for f in mnist.files]
        return X_train, y_train, X_test, y_test


