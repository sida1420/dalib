import random
import copy
def solo(i, j,level, crowding):
        if level[i]==level[j]:
            return i if crowding[i]>crowding[j] else j
        return i if level[i]<level[j] else j

def pathConnect(ind1, ind2):
    i=random.randint(0,len(ind1)-2)
    j=random.randint(1,len(ind2)-1)

    rawPath= ind1[0:i+1]+ind2[j:len(ind2)]
    # return copy.deepcopy(rawPath)
    return [p.copy() for p in rawPath]

import Mutate
def crossover(map,inds,evaluations,popuSize,level, crowding):
    offsprings=[]
    n=len(inds)
    while len(offsprings)<popuSize:
        candidates = random.sample(range(n), 4)

        p1=solo(candidates[0],candidates[1],level,crowding)
        p2=solo(candidates[2],candidates[3],level,crowding)

        child1=pathConnect(inds[p1],inds[p2])
        child2=pathConnect(inds[p2],inds[p1])

        #mutate

        if random.random()<0.5:
            Mutate.mutate(map,child1,0.1)
        if random.random()<0.5:
            Mutate.mutate(map,child2,0.1)
        
        offsprings+=[child1,child2]
    return offsprings
