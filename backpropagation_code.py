import numpy as np
import fusi_code as fc

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    return x * (1.0 - x)

class Neural_Network:
    def __init__(self, input, hidden, output, learningRate):
        self.n_input = input
        self.n_hidden = hidden
        self.n_output = output
        self.learning_rate = learningRate
        self.bias_hidden = np.ones((hidden, 1))
        self.bias_output = np.ones((output, 1))

        self.weight_hidden = np.random.rand(self.n_hidden, self.n_input) - 0.5
        self.weight_output = np.random.rand(self.n_output, self.n_hidden) - 0.5

    def forward(self, input_list):
        inputs = np.array(input_list, ndmin=2).T
        self.hiddenLayer = np.dot(self.weight_hidden, inputs, self.bias_hidden)
        self.Net_hiddenLayer = sigmoid(self.hiddenLayer)

        self.outputLayer = np.dot(self.weight_output, self.Net_hiddenLayer, self.bias_output)
        self.Net_outputLayer = sigmoid(self.outputLayer)
        return self.Net_outputLayer

    def calc_loss(self, target_list):
        targets = np.array(target_list, ndmin=2).T
        m = targets.shape[0]
        cost = -(1.0 / m) * (np.dot(target_list, np.log(self.Net_outputLayer)) + np.dot(1 - target_list, np.log(1 - self.Net_outputLayer)))
        return cost

    def backprop(self, input_list, target_list):
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        self.output_error = targets - self.Net_outputLayer
        self.output_error_weight = np.dot(self.output_error * deriv_sigmoid(self.Net_outputLayer), self.Net_hiddenLayer.T)
        self.output_error_bias = np.sum(self.output_error, axis=1, keepdims=True)

        self.hidden_error = np.dot(self.weight_output.T, self.output_error)
        self.hidden_error_weight = np.dot(self.hidden_error * deriv_sigmoid(self.Net_hiddenLayer), inputs.T)
        self.hidden_error_bias = np.sum(self.hidden_error, axis=1, keepdims=True)

    def update_params(self):
        self.weight_output += self.learning_rate * self.output_error_weight
        self.weight_hidden += self.learning_rate * self.hidden_error_weight
        self.bias_output += self.learning_rate * self.output_error_bias
        self.bias_hidden += self.learning_rate * self.hidden_error_bias

    def predict(self, input_list):
        inputs = np.array(input_list, ndmin=2).T
        self.hiddenLayer = np.dot(self.weight_hidden, inputs, self.bias_hidden)
        self.Net_hiddenLayer = sigmoid(self.hiddenLayer)

        self.outputLayer = np.dot(self.weight_output, self.Net_hiddenLayer, self.bias_output)
        self.Net_outputLayer = sigmoid(self.outputLayer)
        return self.Net_outputLayer

    def train_data(self, data_train, epochs):
        accu_train = []
        for i in range(epochs):
            loss = 0
            for record in data_train:
                all_values = record.split(',')
                inputs = fc.convert_decimal(fc.add_dot_separator(all_values[1:]))
                targets = np.zeros(n.n_output) + 0.01
                targets[int(all_values[0])] = 0.99
                correct_label = int(all_values[0])
                n.forward(inputs)
                loss = n.calc_loss(targets)
                n.backprop(inputs, targets)
                n.update_params()
                outputs = n.predict(inputs)
                label = np.argmax(outputs)
                if (label == correct_label):
                    accu_train.append(1)
                else:
                    accu_train.append(0)
            if i % 10 == 0:
                print("Epochs = ", i)
                print("Cross-Entropy Loss =", loss)
                print("Loss =", np.mean(np.square(targets - n.forward(inputs))))
        accu_array = np.asarray(accu_train)
        print("Accu Train: ", accu_array.sum() / accu_array.size * 100, "%")

train_data_file = open('new_aircraft_fusi.csv', 'r')
train_data_list = train_data_file.readlines()
train_data_file.close()

n = Neural_Network(input=1, hidden=100, output=2, learningRate=0.1)
n.train_data(train_data_list, 10)
test_data_file = open('test_fusi.csv', 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

scoreboard = []
for i in test_data_list:
    all_values = i.split(',')
    correct_label = int(all_values[0])
    inputs = fc.convert_decimal(fc.add_dot_separator(all_values[1:]))
    outputs = n.predict(inputs)
    label = np.argmax(outputs)
    if(label == correct_label):
        scoreboard.append(1)
    else:
        scoreboard.append(0)

scorecard_array = np.asarray(scoreboard)
print("Accu Test: ", scorecard_array.sum() / scorecard_array.size * 100, "%")

#test without label
# inputs = fc.convert_decimal(fc.add_dot_separator(['11111011'])) - 1.0
# outputs = n.predict(inputs)
# print(outputs)
#
# label = np.argmax(outputs)
# print(label)