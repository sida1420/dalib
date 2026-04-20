import random
import math

combinations=None
permutations=None

def precompute(n):
    global combinations, permutations
    permutations=[0]*(n+1)
    permutations[0]=1

    for i in range(1,n+1):
        permutations[i]=i*permutations[i-1]

    combinations=[list()]*(n+1)

    for i in range(0,n+1):
        combinations[i]=[0]*(i+1)
        for j in range(0,i+1):
            combinations[i][j]=permutations[i]/(permutations[j]*permutations[i-j])
# 0.3, 0.5, 0.1 len=3
# 0,   1,    2

# c[1]*0.3*x**1*(1-x)**(4-1)



def func(ind, x):
    
    n=len(ind)+1
    ans=(x**n)
    #1, 2, 3,.., n-1
    for i in range(len(ind)):
        k=i+1
        ans+=combinations[n][k]*ind[i]*(x**k)*((1-x)**(n-k))
    
    return ans

#linear change find ax+b to make f(1)+a*1+b=1
# def normalize(ind):
#     nor_ind=ind.copy()
#     y1=func(nor_ind,1)
#     nor_ind[0]+=1-y1
#     return nor_ind


def probabilize(ind, sample_size, num_bins):
    step=1/sample_size
    samples=[func(ind, step*i) for i in range(sample_size)]

    bins=[0]*num_bins
    wrong=0
    step=1/num_bins
    for sample in samples:
        if sample<0 or sample>=1:
            wrong+=1
        else:
            idx=int(sample/step)
            if idx>=num_bins:
                idx-=1
            bins[idx]+=1

    s=sum(bins)
    if s==0:
        return bins
    for bin in bins:
        bin/=s

    return bins


from Classes import Point

def averagilize(iterable):
    avg=sum(iterable)/len(iterable)

    if avg==0:
        return iterable

    return [x/avg for x in iterable]

def different(bins, target):
    avg_bin=averagilize(bins)
    avg_target_y=averagilize([p.y for p in target])
    diffs=[]
    for i in range(len(avg_target_y)):
        idx=int(target[i].x*len(bins))
        if idx==len(bins):
            idx-=1
        diffs.append(avg_target_y[i] - avg_bin[idx])


    mse = math.sqrt(sum([d**2 for d in diffs]) / len(diffs))

    return mse

def complexity(ind):
    return len(ind)


#likely wrong
def steepness(bins):
    
    diffs=[(bins[i]-bins[i-1])**2 for i in range(1, len(bins))]

    if len(diffs)==0:
        return 0

    return math.sqrt(sum(diffs)/len(diffs))

def error(ind, num_divisions):
    ans=0
    step=1/num_divisions
    pre=func(ind,0)
    for i in range(0,num_divisions+1):
        cur=func(ind,step*i)
        if cur<0:
            ans-=cur
        if cur>1:
            ans+=cur-1

        if cur<pre:
            ans+=abs(cur-pre)
        pre=cur

    return ans/num_divisions


def evaluate(population, target):
    nor_population=population
    distributed_population=[probabilize(ind,1000,50) for ind in nor_population]

    evas=[]

    for i in range(len(population)):
        eva={"different":different(distributed_population[i],target), "complexity":complexity(nor_population[i]), "steepness":steepness(distributed_population[i]), "errors": error(nor_population[i],100)}
        evas.append(eva)

    return evas

# target=[Point(0,0.5),Point(0.5,0.5),Point(0.6,0.5),Point(1,0.5)]
# print(evaluate([[1,0,0,0,0,0,0,0,0,0],],target))

from Vector import Vector
def gbip(weight, eva, reference, penalty, objectives):
    v_eva=Vector([eva[obj] for obj in objectives])
    nor_weight=weight/abs(weight)

    fz=reference-v_eva
    d1=abs(fz*nor_weight)/abs(nor_weight)
    d2=abs(v_eva - (reference + nor_weight*d1))

    return d1 + penalty*d2
    
def update_ref(evas, reference,objectives):
    for eva in evas:
        for i in range(len(objectives)):
            obj=objectives[i]
            reference.vals[i]=min(reference.vals[i],eva[obj])
    return reference
    