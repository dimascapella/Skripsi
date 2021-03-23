a = '0'
b = '010101010101010101001'

data = list(a + b)
data.insert(1, ',')
new_value = ''.join(data)
print(new_value)