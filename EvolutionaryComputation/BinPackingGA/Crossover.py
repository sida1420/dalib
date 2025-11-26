
from collections import defaultdict
def cycleCrossover(parent1, parent2, start):
    child=[0 for i in range(len(parent1))]
    child[start]=parent1[start]
    ha=defaultdict(list)
    for i in range(len(parent2)):
        ha[parent2[i]].append(i)
    current=ha[parent1[start]].pop()
    while current!=start:
        child[current]=parent1[current]
        current=ha[parent1[current]].pop()
    i=start+1
    while i<len(parent1) and child[i]!=0:
        i+=1
    if i>=len(parent1):
        return child
    # print(child)
    subChild=cycleCrossover(parent2,parent1,i)
    for i in range(len(parent1)):
        if child[i]==0:
            child[i]=subChild[i]
    return child


def crossover(gens, percent):        
    newGens=[]
    n=int(len(gens)*percent)-1
    for i in range(0,n,2):
        newGens.append(cycleCrossover(gens[i],gens[i+1],0))
        newGens.append(cycleCrossover(gens[i+1],gens[i],0))
    # print(newGens)
    return newGens