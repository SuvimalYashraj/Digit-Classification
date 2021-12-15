import random
from copy import deepcopy
import numpy as np
from layer import Input_Layer, Hidden_Layer, Activation, Final_Layer

class Neural_Network():

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network. """
        biases = [np.random.randn(y, 1) for y in sizes[1:]]
        weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.neural_network = [Input_Layer()]

        for layer in range(1,len(sizes)):
            self.neural_network.append(Hidden_Layer(weights[layer-1],biases[layer-1]))
            if layer == len(sizes)-1:
                break
            self.neural_network.append(Activation())    
        self.neural_network.append(Final_Layer())


    def SGD(self, training_data, epochs, batch_size, eta):
       
        training_zip = list(zip(training_data[0],training_data[1]))
        number_train_images = np.shape(training_data[0])[0]

        for j in range(epochs):
            correct_predictions = 0
            loss = 0

            random.shuffle(training_zip)
            training_data_shuffled = list(zip(*training_zip))
            batches_disjointed = ([training_data_shuffled[0][i:i+batch_size] for i in range(0, number_train_images, batch_size)],[training_data_shuffled[1][i:i+batch_size] for i in range(0, number_train_images, batch_size)])
            batches = list(zip(batches_disjointed[0],batches_disjointed[1]))

            for batch in batches:
                self.forward_propagation(batch)
                self.backward_propagation(batch,eta)

                prediction = np.argmax(self.neural_network[-1].output,axis=1)
                original = np.argmax(np.asarray(batch[1]),axis=1)
                correct_predictions += np.sum(prediction==original)
                
                cross_entropy_loss = self.cross_entropy(np.asarray(batch[1]),self.neural_network[-1].output)
                loss += np.sum(cross_entropy_loss,axis=0,keepdims=True)
                
            print ("Epoch {0}: {1} {2}".format(j, correct_predictions, loss))

    def predict(self, validation):
        self.forward_propagation(validation,1)
        prediction = np.argmax(self.neural_network[-1].output,axis=1)
        # original = np.argmax(np.asarray(validation[1]),axis=1).astype(int)
        # correct_predictions = np.sum(prediction==original)
        # print(correct_predictions)
        return prediction

    def forward_propagation(self, batch, testing=0):
        if testing:
            image = np.asarray(batch)
        else:
            image = np.asarray(batch[0])
        image = image/255
        self.neural_network[0].set_input(image)

        for layer in range(1,len(self.neural_network)):
            self.neural_network[layer].feed_forward(self.neural_network[layer-1].output)

    def backward_propagation(self, batch, eta):
        label = np.asarray(batch[1])
        self.neural_network[-1].back_propagation(label)

        for layer in range(len(self.neural_network)-2,0,-1):
            self.neural_network[layer].back_propagation(self.neural_network[layer+1].gradient,eta)

    # calculate cross entropy
    def cross_entropy(self, label, predicted):
        log_predictions = np.log2(predicted)
        label_product = label * log_predictions
        loss = np.sum(label_product,axis=1,keepdims=True)
        loss = loss * -1
        return loss