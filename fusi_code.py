import numpy as np
arrays = ['0000000000000001', '0000000000100000', '0000001000000000', '0100000000000000', '0000010000100000', '0000001001000000', '0000000100000011', '0000000010001000', '0010001000000000', '1000100000000000', '1001000100000010', '1000100010000000']
coba = ['0000000000000001', '0000000000100000']

array = [['0000000000000001', '0000000000100000', '0000010000000000', '0100001000000000'],
         ['0100000000000000', '0000010000100000', '0000001001000000', '0000010000100000'],
         ['0100000000000000', '0000010000100000', '0000001001000000', '0000010000100000'],
         ['0100000000000000', '0000010000100000', '0000001001000000', '0000010000100000'],
         ['0100000000000000', '0000010000100000', '0000001001000000', '0000010000100000']]


wings = []

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def XOR(x, y):
    if x != y:
        return '1'
    else:
        return '0'

def add_dot_separator(binary):
    if len(binary) <= 1:
        for all_values in range(len(binary)):
            new_value = list(binary[all_values])
        new_value.insert(2, '.')
        new_value = ''.join(new_value).split()
    elif len(binary) > 1:
        new_value = list(binary)
        new_value.insert(2, '.')
        new_value = ''.join(new_value).split()
    return new_value

def convert(binary):
    decimal = ''
    value = 0
    for i in range(len(binary)):
        decimal = binary[0]
    decimal = list(decimal)
    size = len(decimal)
    while i < len(decimal):
        value += int(decimal[size - 1 - i]) * pow(2, i)
        i += 1
    return value

def convert_decimal(binary):
    decimal = ''
    positif = negatif = 0
    value = 0.0
    for i in range(len(binary)):
        decimal = binary[0]
    decimal = list(decimal)
    index_dot = decimal.index('.')
    size_dot = decimal.index('.')
    for i in range(len(decimal)):
        if i < index_dot:
            positif = positif + 1
            if decimal[i] == '1':
                value = value + pow(2, (size_dot - 1))
        elif i > index_dot:
            negatif = negatif + 1
            if decimal[i] == '1':
                value = value + pow(2, -negatif)
        size_dot -= 1
    return value

def _fusi(array):
    for i in range(len(array) - 1):
        temp = []
        for j in range(len(array[0])):
            if i == 0:
                temp += XOR(array[i][j], array[i + 1][j])
            else:
                temp += XOR(information_fusion[0][j], array[i + 1][j])
        information_fusion = ''.join(temp).split()
    return information_fusion

def batch_arrays(array_data, array_lisy_variable):
    fusi = []
    for i in range(len(array_data)):
        for j in range(len(array_data[0]) - 1):
            temp = []
            for k in range(len(array_data[0][0])):
                if j == 0:
                    temp += XOR(array_data[i][j][k], array_data[i][j + 1][k])
                else:
                    temp += XOR(fusi[0][k], array_data[i][j + 1][k])
            fusi = ''.join(temp).split()
        array_lisy_variable.append(fusi)


# print("Fusion Information Data =", information_fusion)
print("Preprocessing Data =", add_dot_separator(['0100111111100000']))
print("Preprocessing Data =", convert_decimal(add_dot_separator(['0000000000000110'])))
# print(convert(['0100111111100000']))
# batch_arrays(array, wings)
# print(wings)