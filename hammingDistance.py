a = ['0000011101100011']
b = [['0001100101011000'],
     ['0000000100001000'],
     ['0001100101011000'],
     ['0001100101011000'],
     ['0000000100001000'],
     ['0000001101100011'],
     ['0001100101011000']]

c = ['1100110101001010']
d = [['0101111000000100'],
     ['0000000100001000'],
     ['0001100101011000'],
     ['0001100101011000'],
     ['0000000100001000'],
     ['1100110101001010'],
     ['0000011101100011'],
     ['0000000100001000']]

new_arrays = []

def hammingDistance(value1, value2):
    distance = 0
    for i in range(len(value1)):
        for j in range(len(value1[0])):
            if value1[i][j] != value2[i][j]:
                distance += 1
    return distance

def get_min_hd(value1, value2):
    min_distance = 0
    value_min_distance = ''
    current_min_distance = hammingDistance(value1, value2[0])
    for i in range(len(value2) - 1):
        for j in range(len(value2[i])):
            for k in range(len(value2[i][j])):
                if value1[j][k] != value2[i + 1][j][k]:
                    min_distance += 1
            if min_distance < current_min_distance:
                current_min_distance = min_distance
                value_min_distance = value2[i + 1][j]
            min_distance = 0
    return value_min_distance

new_arrays.append(get_min_hd(a, b))
new_arrays.append(get_min_hd(c, d))
print(new_arrays)

print(hammingDistance(a, b[0]))