import time
import numpy as np
import random

length = 10000000

a = np.zeros(length)
b = np.zeros(length)

for i in range (0, length):
    a[i] = random.randint(1, 100)

for i in range (0, length):
    b[i] = random.randint(1, 100)

start = time.time()
c = np.dot(a, b)
end = time.time()

print ('vectorised time to run:',end - start,'ms')

c = 0
start = time.time()
for i in range (0, length):
    c = c + a[i] * b[i]
end = time.time()

print ('unvectorised time to run:',end - start,'ms')