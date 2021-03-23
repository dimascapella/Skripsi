arrays = ['0000000000000001', '0000000000100000', '0000001000000000', '0100000000000000', '0000010000100000', '0000001001000000', '0000000100000011', '0000000010001000', '0010001000000000', '1000100000000000', '1001000100000010', '1000100010000000']
information_fusion = []
coba = ['0000000000000001', '0000000000100000']

def XOR(x, y):
    if x != y:
        return '1'
    else:
        return '0'

def add_dot_separator(binary):
    if len(binary) <= 1:
        for all_values in range(len(binary)):
            new_value = list(binary[all_values])
        new_value.insert(1, '.')
        new_value = ''.join(new_value).split()
    elif len(binary) > 1:
        new_value = list(binary)
        new_value.insert(1, '.')
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
    for i in range(len(decimal)):
        index_dot = decimal.index('.')
        if i < index_dot:
            positif = positif + 1
            if decimal[i] == '1':
                value = value + pow(2, (positif - 1))
        elif i > index_dot:
            negatif = negatif + 1
            if decimal[i] == '1':
                value = value + pow(2, -negatif)
    if value > 1.0:
        value = value - 1.0
    value = round(value, 2)
    return value

for i in range(len(arrays) - 1):
    temp = []
    for j in range(len(arrays[0])):
        if i == 0:
            temp += XOR(arrays[i][j], arrays[i + 1][j])
        else:
            temp += XOR(information_fusion[0][j], arrays[i + 1][j])
    information_fusion = ''.join(temp).split()

print("Fusion Information Data =", information_fusion)
print("Preprocessing Data =", convert_decimal(add_dot_separator(['0100111111100000'])))
print(convert(['0100111111100000']))