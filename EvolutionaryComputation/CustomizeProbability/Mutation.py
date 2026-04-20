import random

def jitter(ind):
    choices=[1,-1]
    for i in range(len(ind)):
        if random.random()<0.1:
            ind[i]+=(random.random()**2)*random.choice(choices)/2
    return ind
import Classes
#not work
def simplify(ind):
    for i in range(len(ind)):
        if random.random()<0.1:
            ind[i]+=(-random.random()**2)*ind[i]
    return ind
def mutate(ind):

    
    ind=Classes.repair(jitter(ind))

    return ind

    