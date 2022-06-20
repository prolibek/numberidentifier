import numpy as np

# sigmoid function
def sigmoid(x):
    return 1/(1 + np.exp(-x))
    
class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, 
                 outputnodes, learningrate):
        self.inodes = inputnodes #number of each type of nodess
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # setting weights for input -> hidden layer
        # i didn't unstertand how does that work
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5),
                                      (self.hnodes, self.inodes))
        # eights for hidden -> output
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5),
                                      (self.onodes, self.hnodes))
        
        self.lrate = learningrate
        
        # setting sigmoid as activation function
        self.activation_function = lambda x: sigmoid(x)
        
    def train(self, inputs_list, targets_list):
        # creating arrays of input and target values
        inputs = np.array(inputs_list, ndmin = 2).T
        targets = np.array(targets_list, ndmin = 2).T
        
        # multyplying matrix of weights to matrix of inputs 
        hidden_inputs = np.dot(self.wih, inputs)
        # activation
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        # errors backpropagation
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)
        
        # gradient descent approach
        # weight = learning_rate * sigmoid * (1 - sigmoid) * input_matrix
        self.who += self.lrate * np.dot((output_errors * final_outputs
                                          * (1.0 - final_outputs)), 
                                           np.transpose(hidden_outputs))
        self.wih += self.lrate * np.dot((hidden_errors * hidden_outputs 
                                         * (1.0 - hidden_outputs)), 
                                        np.transpose(inputs))
    
    def query(self, inputs_list):

        # same actions as in train actions
        inputs = np.array(inputs_list, ndmin = 2).T
        
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs