# MNIST-MLP

An implementation of a Multilayered Perceptron to classify the MNIST DataSet
This is an expansion of our most basic model: The Perceptron

In this model, we will use the Mnist handwritten dataset to classify digits that have class labels from 1 - 10

The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. 

The digits have been size-normalized and centered in a fixed-size image.


It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting.

Four files are available on this site:



To Obtain the Mnist Dataset
Use URL:  http://yann.lecun.com/exdb/mnist/

And consists the following parts:

- train-images-idx3-ubyte.gz:  training set images (9912422 bytes)
- train-labels-idx1-ubyte.gz:  training set labels (28881 bytes)
- t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes)
- t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)

Recommend to unzip using the Unix/Linux gzip tool from the Terminal for efficiency, 
using the following command in your local MNIST download directory:

gzip *ubyte.gz -d
