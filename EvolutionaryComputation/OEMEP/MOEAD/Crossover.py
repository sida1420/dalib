import random
import copy

def pathConnect(ind1, ind2):
    i=random.randint(0,len(ind1)-2)
    j=random.randint(1,len(ind2)-1)

    rawPath= ind1[0:i+1]+ind2[j:len(ind2)]
    # return copy.deepcopy(rawPath)
    return [p.copy() for p in rawPath]

import Mutate
def crossover(map, population,neighbours,popuSize):
    offsprings=[]
    

    for idx in range(popuSize):
        p1=random.randint(0,len(neighbours[idx])-1)
        p2=random.randint(0,len(neighbours[idx])-1)

        child=pathConnect(population[neighbours[idx][p1]],population[neighbours[idx][p2]])

        if random.random()<0.5:
            Mutate.mutate(map,child,0.1)

        offsprings.append(child)
    return offsprings
