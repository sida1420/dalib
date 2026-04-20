
import random

import Classes
import Mutation

def share_branch(p1, p2, max_complexity):
    n1=p1.pre_size+1
    n2=p2.pre_size+1
    child=p1.copy()
    
    options1=[node for node in child.get_list() if node.oper is not None]
    i=random.randint(0,len(options1)-1)
    cut=options1[i]
    # print(cut)
    k=0 if cut.oper in Classes.one else random.choice([0,1])
    
    available=max_complexity-n1+cut.pre[k].pre_size+1
    # print(available)
    options2=[node for node in p2.get_list() if node.pre_size<available]

    j=random.randint(0,len(options2)-1)
    cut.pre[k]=options2[j].copy()
    child.update_pre()
    return child


    
def crossover(population, neighbours,max_complexity):

    offsprings=[]
    

    for idx in range(len(population)):
        p1i=random.randint(0,len(neighbours[idx])-1)
        p2i=random.randint(0,len(neighbours[idx])-1)
        p1=population[neighbours[idx][p1i]]
        p2=population[neighbours[idx][p2i]]

        child=share_branch(p1,p2,max_complexity)

        if random.random()<0.5:
            child=Mutation.mutate(child,max_complexity)

        offsprings.append(child)
    return offsprings

