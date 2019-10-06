import numpy as np
from fileLoader import load_file
import matplotlib.pyplot as plt

df = load_file('mnist_data.npz')
X_train, y_train, X_test, y_test = df.get_data()


print(X_train.shape)


fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(10):
    img = X_train[y_train==i][0].reshape(28,28)
    ax[i].imshow(img, cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(25):
    img = X_train[y_train == 6][i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')


ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
