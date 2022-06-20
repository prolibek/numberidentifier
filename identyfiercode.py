from neuralnetwork import *
import numpy as np
import matplotlib.pyplot

training_data_file = open("dataset/mnist_train.csv", 'r')
training_data_list  = training_data_file.readlines()
training_data_file.close()

testing_data_file = open("dataset/mnist_test.csv", 'r')
testing_data_list = testing_data_file.readlines()
testing_data_file.close()

input_nodes = 784
hidden_nodes = 100
output_nodes = 10

learning_rate = 0.3

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes,
                learning_rate)

EPOCHES = 5
for i in range(EPOCHES):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = np.zeros(output_nodes)

        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
    print("Epoch #", i+1, " is passed", sep = "")

scorecard = []
for record in testing_data_list:
    all_values = record.split(',')

    correct_answer = int(all_values[0])

    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

    outputs = n.query(inputs)

    answer = np.argmax(outputs)

    if answer == correct_answer:
        scorecard.append(1)
    else:
        scorecard.append(0)

print(scorecard)
print("Efficiency", sum(scorecard) / len(scorecard) * 100, "%")