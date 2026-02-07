
import random
def random_passing(p1, p2):
    child=[]
    for i in range(len(p1)):
        if random.random()<0.5:
            child.append(p1[i])
        else:
            child.append(p2[i])
    return child.copy()

def SBX(p1, p2, nc):
    child=[]
    for i in range(len(p1)):
        r=random.random()

        if r<=0.5:
            b=(2*r)**(1/(nc+1))
        else:
            b=(1/2/(1-r))**(1/(nc+1))

        if random.random()<0.5:
            b=-b
        child.append(0.5*(p1[i]+p2[i]+b*(p2[i]-p1[i])))
    return child

import Mutation
def crossover(population, neighbours):

    offsprings=[]
    

    for idx in range(len(population)):
        p1=random.randint(0,len(neighbours[idx])-1)
        p2=random.randint(0,len(neighbours[idx])-1)

        child=SBX(population[neighbours[idx][p1]],population[neighbours[idx][p2]],1)
        # child=population[neighbours[idx][p1]].copy()

        if random.random()<0.5:
            child=Mutation.mutate(child)

        offsprings.append(child)
    return offsprings

