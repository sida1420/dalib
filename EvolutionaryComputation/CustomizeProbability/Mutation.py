import random

def jitter(ind):
    choices=[1,-1]
    for i in range(len(ind)):
        if random.random()<0.1:
            ind[i]+=(random.random()**2)*random.choice(choices)
    return ind

def simplify(ind):
    for i in range(len(ind)):
        if random.random()<0.1:
            ind[i]+=(-random.random()**2)*ind[i]
    return ind
def mutate(ind):

    if random.random()<0.5:
        ind=simplify(ind)
    else:
        ind=jitter(ind)

    return ind

    