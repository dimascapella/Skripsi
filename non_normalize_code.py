import numpy as np
import time
import fusi_code as fc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    return (x) * (1.0 - (x))

class Neural_Network:
    def __init__(self, input, hidden, output, learningRate):
        self.loss_attemp = []
        self.n_input = input
        self.n_hidden = hidden
        self.n_output = output
        self.learning_rate = learningRate
        self.bias_hidden = np.ones((hidden, 1))
        self.bias_output = np.ones((output, 1))

        self.weight_hidden = np.random.uniform(low=-0.2, high=0.2, size=(self.n_hidden, self.n_input))
        self.weight_output = np.random.uniform(low=-0.2, high=0.2, size=(self.n_output, self.n_hidden))

    def forward(self, input_list):
        inputs = np.array(input_list, ndmin=2).T
        self.hiddenLayer = np.dot(self.weight_hidden, inputs) + self.bias_hidden
        self.Net_hiddenLayer = sigmoid(self.hiddenLayer)

        self.outputLayer = np.dot(self.weight_output, self.Net_hiddenLayer) + self.bias_output
        self.Net_outputLayer = sigmoid(self.outputLayer)
        return self.Net_outputLayer

    def calc_loss(self, target_list):
        targets = np.array(target_list, ndmin=2).T
        m = targets.shape[0]
        cost = 0
        # cost = -(1.0 / m) * (np.dot(target_list, np.log(self.Net_outputLayer)) + np.dot(1 - target_list, np.log(1 - self.Net_outputLayer)))
        cost = cost + ((self.Net_outputLayer - targets) ** 2).sum() / m
        return cost

    def backprop(self, input_list, target_list):
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T
        m = targets.shape[0]

        self.output_error = self.Net_outputLayer - targets
        self.output_error_weight = np.dot(self.output_error * deriv_sigmoid(self.Net_outputLayer),
                                          self.Net_hiddenLayer.T) * (1 / m)
        self.output_error_bias = np.sum(self.output_error, axis=1, keepdims=True) * (1 / m)

        self.hidden_error = np.dot(self.weight_output.T, self.output_error)
        self.hidden_error_weight = np.dot(self.hidden_error * deriv_sigmoid(self.Net_hiddenLayer), inputs.T) * (1 / m)
        self.hidden_error_bias = np.sum(self.hidden_error, axis=1, keepdims=True) * (1 / m)

    def update_params(self):
        self.weight_output = self.weight_output - self.learning_rate * self.output_error_weight
        self.weight_hidden = self.weight_hidden - self.learning_rate * self.hidden_error_weight
        self.bias_output = self.bias_output - self.learning_rate * self.output_error_bias
        self.bias_hidden = self.bias_hidden - self.learning_rate * self.hidden_error_bias

    def predict(self, input_list):
        inputs = np.array(input_list, ndmin=2).T
        self.hiddenLayer = np.dot(self.weight_hidden, inputs) + self.bias_hidden
        self.Net_hiddenLayer = sigmoid(self.hiddenLayer)

        self.outputLayer = np.dot(self.weight_output, self.Net_hiddenLayer) + self.bias_output
        self.Net_outputLayer = sigmoid(self.outputLayer)
        return self.Net_outputLayer

    def train_data(self, data_train, epochs):
        accu_train = []
        all_label = []
        for i in range(epochs):
            loss = 0
            startTime = time.time()
            for record in data_train:
                all_values = record.split(',')
                inputs = (np.asfarray(all_values[1:]))
                targets = np.zeros(n.n_output) + 0.01
                targets[int(all_values[0])] = 0.99
                correct_label = int(all_values[0])
                n.forward(inputs)
                loss = n.calc_loss(targets)
                n.backprop(inputs, targets)
                n.update_params()
                outputs = n.predict(inputs)
                label = np.argmax(outputs)
                all_label.append(label)
                if (label == correct_label):
                    accu_train.append(1)
                else:
                    accu_train.append(0)
            if i % 10 == 0:
                print("Epochs = ", i)
                print("Loss =", loss)
            self.loss_attemp.append(loss)
        accu_array = np.asarray(accu_train)
        self.accu = accu_array.sum() / accu_array.size * 100
        print("Accu Train: ", self.accu , "%")
        print("Last Loss =", loss)
        print('The script took {0} second !'.format(time.time() - startTime))

train_data_file = open('new_dataset_separate_fusi.csv', 'r')
train_data_list = train_data_file.readlines()
train_data_file.close()

new_data = []
for i in train_data_list:
    data = i.split(',')
    classes = list(str(data[0]))
    classes.insert(1, ',')
    classes = ''.join(classes)
    wings = list(str(fc.convert_decimal(fc.add_dot_separator(data[1]))))
    wings.insert(17, ',')
    wings = ''.join(wings)
    engines = list(str(fc.convert_decimal(fc.add_dot_separator(data[2]))))
    engines.insert(17, ',')
    engines = ''.join(engines)
    fuselages = list(str(fc.convert_decimal(fc.add_dot_separator(data[3]))))
    fuselages.insert(17, ',')
    fuselages = ''.join(fuselages)
    tails = list(str(fc.convert_decimal(fc.add_dot_separator(data[4]))))
    tails.insert(17, ',')
    tails = ''.join(tails)
    additionals = str(fc.convert_decimal(fc.add_dot_separator(data[5])))
    data_normalize = classes + wings + engines + fuselages + tails + additionals
    new_data.append(data_normalize)

n = Neural_Network(input=5, hidden=300, output=4, learningRate=0.6)
n.train_data(train_data_list, 1000)

test_data = open('separate_fusi_test.csv','r')
test_data_list = test_data.readlines()
test_data.close()

actual = []
predict = []
new_test_data = []
for i in test_data_list:
    data = i.split(',')
    classes = list(str(data[0]))
    classes.insert(1, ',')
    classes = ''.join(classes)
    wings = list(str(fc.convert_decimal(fc.add_dot_separator(data[1]))))
    wings.insert(17, ',')
    wings = ''.join(wings)
    engines = list(str(fc.convert_decimal(fc.add_dot_separator(data[2]))))
    engines.insert(17, ',')
    engines = ''.join(engines)
    fuselages = list(str(fc.convert_decimal(fc.add_dot_separator(data[3]))))
    fuselages.insert(17, ',')
    fuselages = ''.join(fuselages)
    tails = list(str(fc.convert_decimal(fc.add_dot_separator(data[4]))))
    tails.insert(17, ',')
    tails = ''.join(tails)
    additionals = str(fc.convert_decimal(fc.add_dot_separator(data[5])))
    data_normalize = classes + wings + engines + fuselages + tails + additionals
    new_test_data.append(data_normalize)

# for i in test_data_list:
#     all_values = i.split(',')
#     input_data = (np.asfarray(all_values[1:]))
#     target_data = np.zeros(n.n_output) + 0.01
#     target_data[int(all_values[0])] = 0.99
#     actual.append(int(all_values[0]))
#     outputs = n.predict(input_data)
#     predict.append(np.argmax(outputs))

random_data = open('data_uji_random.csv', 'r')
random_data_list = random_data.readlines()
random_data.close()

actual = [0,0,0,1,1,2,2,2,3,3]
predict = []
# new_test_data = []
# for i in random_data_list:
#     data = i.split(',')
#     # classes = list(str(data[0]))
#     # classes.insert(1, ',')
#     # classes = ''.join(classes)
#     wings = list(str(fc.convert_decimal(fc.add_dot_separator(data[0]))))
#     wings.insert(17, ',')
#     wings = ''.join(wings)
#     engines = list(str(fc.convert_decimal(fc.add_dot_separator(data[1]))))
#     engines.insert(17, ',')
#     engines = ''.join(engines)
#     fuselages = list(str(fc.convert_decimal(fc.add_dot_separator(data[2]))))
#     fuselages.insert(17, ',')
#     fuselages = ''.join(fuselages)
#     tails = list(str(fc.convert_decimal(fc.add_dot_separator(data[3]))))
#     tails.insert(17, ',')
#     tails = ''.join(tails)
#     additionals = str(fc.convert_decimal(fc.add_dot_separator(data[4])))
#     data_normalize = wings + engines + fuselages + tails + additionals
#     new_test_data.append(data_normalize)
#
for i in random_data_list:
    all_values = i.split(',')
    input_data = np.asfarray(all_values[0:])
    print(input_data)
    outputs = n.predict(input_data)
    predict.append(np.argmax(outputs))

print(actual)
print(predict)

print(confusion_matrix(actual, predict))
print(classification_report(actual, predict))

# plot1 = plt.figure(1)
# plt.plot(n.loss_attemp)
# plt.show()
# classes_data = []
# wings_data = []
# engine_data = []
# fuselage_data = []
# tail_data = []
# additional_data = []
# wings_fusi = []
# engine_fusi = []
# additional_fusi = []
# data = []
# for record in train_data_list:
#     all_values = record.split(',')
#     classes_data.append(all_values[0].split())
#     wings_data.append(all_values[1:5])
#     engine_data.append(all_values[5:8])
#     fuselage_data.append(all_values[8].split())
#     tail_data.append(all_values[9].split())
#     additional_data.append(all_values[10:14])
# fc.batch_arrays(wings_data, wings_fusi)
# fc.batch_arrays(engine_data, engine_fusi)
# fc.batch_arrays(additional_data, additional_fusi)
# for i in range(len(wings_fusi)):
#     new_value = classes_data[i] + wings_fusi[i] + engine_fusi[i] + fuselage_data[i] + tail_data [i] + additional_fusi[i]
#     data.append(new_value)
#
# for i in range(len(classes_data)):
#     print(data[i][5])

# test without label
# inputs1 = np.asfarray(['0100001000100001','0000011101100011','0000000010001000','0010001000000000','1100110101001010'])
# outputs1 = n.predict(inputs1)
# print(outputs1)
#
# label1 = np.argmax(outputs1)
# print(label1)
#
# inputs2 = np.asfarray(['0000100001010010','0001100101011000','0000000100001000','0000000100010000','0101111000000100'])
# outputs2 = n.predict(inputs2)
# print(outputs2)
#
# label2 = np.argmax(outputs2)
# print(label2)

#Uji
#Node Hidden Layer 100 - 300
#LR = 0.6

# wings = []
# engine = []
# fuselage = []
# tail = []
# additional = []
# data_predict = []
# #
# wings.extend(['0000000000000001','0000000000100000','0000001000000000','0100000000000000'])
# engine.extend(['0000010000100000','0000001001000000','0000000100000011'])
# fuselage.extend(['0000000010001000'])
# tail.extend(['0000000100010000'])
# additional.extend(['0100010000000000','1001000100000010','1000100010000000','0011100000000000'])
#
# data_predict.extend([fc._fusi(wings)[0], fc._fusi(engine)[0], fuselage[0], tail[0], fc._fusi(additional)[0]])
# data_input_predict = np.asfarray(data_predict)
# outputs2 = n.predict(data_input_predict)
# print(outputs2)
#
# label2 = np.argmax(outputs2)
# print(label2)

# inputs1 = np.asfarray(['0100001000100001','0000011101100011','0000000010001000','0010001000000000','1100110101001010'])
# outputs1 = n.predict(inputs1)
# print(outputs1)