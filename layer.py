import numpy as np

class Input_Layer:
    def set_input(self, input):
        self.output = input

class Hidden_Layer:
    def __init__(self, weights, bias) -> None:
        self.weights = weights
        self.bias = bias
        self.input = None
        self.output = None
        self.gradient = None

    # matrix multiplication of input with the weights and addition of bias
    def feed_forward(self, input):
        self.input = input
        self.output = (self.input @ self.weights.T) + self.bias.T

    def back_propagation(self, propagated_derivatives, eta):
        self.gradient = propagated_derivatives @ self.weights 

        d_loss_d_weights = propagated_derivatives.T @ self.input
        self.weights -= eta * d_loss_d_weights

        d_loss_d_bias = propagated_derivatives.T
        d_loss_d_bias_avg =  np.sum(d_loss_d_bias,axis=1,keepdims=True)
        self.bias -= eta * d_loss_d_bias_avg

class Activation:
    def __init__(self) -> None:
        self.input = None
        self.output = None
        self.gradient = None

    # sigmoid function
    def feed_forward(self, input):
        self.input = input
        # self.output = np.where(input > 0, input, 0)
        self.output = 1/(1 + np.exp(-input))

    def back_propagation(self, propagated_derivatives, eta=None):
        derivative = self.output * (1-self.output)
        # derivative = np.where(self.input > 0, 1, 0)
        self.gradient = propagated_derivatives * derivative 

class Final_Layer:
    def __init__(self) -> None:
        self.input = None
        self.output = None
        self.gradient = None

    # softmax function
    def feed_forward(self, input):
        self.input = input
        max = np.max(self.input,axis=1,keepdims=True) #returns max of each row and keeps same dims
        e_x = np.exp(self.input - max) #subtracts each row value with its max value
        sum = np.sum(e_x,axis=1,keepdims=True) #returns sum of each row and keeps same dims
        f_x = e_x / sum 
        self.output = f_x

    def back_propagation(self, label):
        derivative = self.output - label 
        N, _ = label.shape
        self.gradient = derivative/N