import sys
import read_write
from neural_network import Neural_Network
from numpy import savetxt

if __name__=='__main__':

    # argument = sys.argv
    # train_image_csv, train_label_csv, test_image_csv =  argument[1], argument[2], argument[3]
    
    # train, test = read_write.load_data(train_image_csv, train_label_csv, test_image_csv)
    train, test = read_write.load_data("train_image.csv", "train_label.csv", "test_image.csv")

    network = Neural_Network([784,100,10])
    network.SGD(train, 20, 10, .12)

    prediction = network.predict(test)
    with open("test_predictions.csv","w") as test_predictions:
        savetxt(test_predictions, prediction, fmt = "%d", delimiter=",")


    # pickle_off = open ("train.txt", "rb")
    # train = pickle.load(pickle_off)
    # pickle_off1 = open ("validate.txt", "rb")
    # validate = pickle.load(pickle_off1)
    # pickle_off1 = open ("test.txt", "rb")
    # test = pickle.load(pickle_off1)
    # with open("train.txt", "wb") as fp:
    #     pickle.dump(train, fp, protocol=pickle.HIGHEST_PROTOCOL)
    # with open("validate.txt", "wb") as fp:
    #     pickle.dump(validate, fp, protocol=pickle.HIGHEST_PROTOCOL)
    # with open("test.txt", "wb") as fp:
    #     pickle.dump(test, fp, protocol=pickle.HIGHEST_PROTOCOL)