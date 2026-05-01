

from sbyte import *
from random import random
def rand_noise(data, p):
    
    for i in range(len(data)):
        for j in range(8):
            if random()<=p:
                data[i][j]=(data[i][j]+1)%2

    return data
