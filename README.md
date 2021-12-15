# Digit Classification
Implementedd a multi-layer perceptron (MLP) neural network to classify hand-written digits without using any machine learning libraries (including
tensorflow (v1&v2), caffe, pytorch, torch, cxxnet, and mxnet). Implemented feedforward/backpropagation as well as training process from scratch.

## Data Description
Used MNIST dataset which consists of four files:
1. Training set images, which contains 60,000 28 × 28 grayscale training images, each representing a single handwritten digit.
2. Training set labels, which contains the associated 60,000 labels for the training images.
3. Test set images, which contains 10,000 28 × 28 grayscale testing images, each representing a single handwritten digit.
4. Test set labels, which contains the associated 10,000 labels for the testing images.
File 1 and 2 are the training set. File 3 and 4 are the test set. Each training and test instance in the MNIST database consists of a 28 × 28 grayscale image of a handwritten digit and an associated integer label indicating the digit that this image represents (0-9). Each of the 28 × 28 = 784 pixels of each of these images is represented by a single 8-bit color channel. Thus, the values each pixel can take on range from 0 (completely black) to 255 (28 − 1, completely white).

## Objective
Implement a multi-hidden-layer neural network learner, that will
(1) Construct a neural network classifier from the given labeled training data,
(2) Use the learned classifier to classify the unlabeled test data, and
(3) Output the predictions of your classifier on the test data into a file in the same directory,
(4) Finish in 30 minutes (for both training your model and making predictions