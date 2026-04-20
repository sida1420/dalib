
import random
def binary_connect(p1, p2, max_complexity):
    i=random.randint(0,min(max_complexity-2,len(p1)-1))
    j=random.randint(min(len(p2)-1,max(0,len(p2)-(max_complexity-i-1))),len(p2)-1)

    child=p1[:i+1].copy()+p2[j:].copy()

    return child

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
import Classes
import Mutation
def crossover(population, neighbours,max_complexity):

    offsprings=[]
    

    for idx in range(len(population)):
        p1i=random.randint(0,len(neighbours[idx])-1)
        p2i=random.randint(0,len(neighbours[idx])-1)
        p1=population[neighbours[idx][p1i]]
        p2=population[neighbours[idx][p2i]]

        r=random.random()
        l=1

        if len(p1)==len(p2):
            l=0.1
        if r<=l:
            child=binary_connect(p1,p2,max_complexity)
        else:
            child=Classes.repair(SBX(p1,p2,1))

        if random.random()<0.5:
            child=Mutation.mutate(child)

        offsprings.append(child)
    return offsprings

