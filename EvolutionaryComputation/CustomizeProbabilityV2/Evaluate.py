import random
import math
from Classes import Operation




def normalize(ind):
    p0=ind(0)
    p1=ind(1)
    nor_ind=ind.copy()
    if p0!=0 and p1!=1:
        return Operation('+', [
            ind, 
            Operation('+', [
                # Part 1: (1 - x) * (-p0)
                Operation('*', [
                    Operation('-', [Operation(None, value=1), Operation(None)]),
                    Operation('neg', [Operation(None,value=p0)])
                ]),
                # Part 2: x * (1 - p1)
                Operation('*', [
                    Operation(None), 
                    Operation('-', [Operation(None, value=1), Operation(None,value=p1)])
                ])
            ])
        ])
    if p0!=0:
        return Operation('+', [
            ind, 
            Operation('*', [
                Operation('-', [Operation(None, value=1), Operation(None)]),
                Operation('neg', [Operation(None,value=p0)])
            ])
        ])
    if p1!=1:
        return Operation('+', [
            ind, 
            Operation('*', [
                Operation(None), 
                Operation('-', [Operation(None, value=1), Operation(None,value=p1)])
            ])
        ])

    return nor_ind

def gradient_error(bins, target):

    avg_bin=averagilize(bins)
    avg_target_y=averagilize([p.y for p in target])
    error=0
    pre_idx=0
    step=1/len(bins)
    for i in range(1,len(avg_target_y)):
        idx=int(target[i].x*len(bins))
        if idx==len(bins):
            idx-=1
        target_slope=(avg_target_y[i]-avg_target_y[i-1])/(target[i].x-target[i-1].x)
        error+=(target_slope-(avg_bin[idx]-avg_bin[pre_idx])/(idx-pre_idx)/step)**2

        # for j in range(pre_idx,idx):
        #     error+=(target_slope-(avg_bin[j+1]-avg_bin[j])/step)**2

        pre_idx=idx

    return error/(len(avg_target_y)-1)

def probabilize(ind, sample_size, num_bins):
    step=1/sample_size
    samples=[ind(step*i) for i in range(sample_size)]

    bins=[0]*num_bins
    wrong=0
    step=1/num_bins
    for sample in samples:
        if sample is None or sample<0 or sample>=1:
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
    return int((ind.pre_size+1)/10)



def error(ind, num_divisions):
    ans=0
    step=1/num_divisions
    pre=ind(0)
    for i in range(0,num_divisions+1):
        cur=ind(step*i)

        if cur<0:
            ans-=cur
        elif cur>1:
            ans+=cur-1
        
        if cur<pre:
            ans+=abs(cur-pre)
        pre=cur
    return min(ans,1e9)

def fatal_error(ind, num_divisions):
    ans=0
    step=1/num_divisions
    for i in range(0,num_divisions+1):
        cur=ind(step*i)
        if cur is None:
            ans+=10
        elif cur<0:
            ans-=cur
        elif cur>1:
            ans+=cur-1
    return min(ans,1e9)

def evaluate(population, target):
    pre_errors=[fatal_error(ind,100) for ind in population]
    nor_population=[normalize(ind) for ind in population]
    distributed_population=[probabilize(ind,1000,50) for ind in nor_population]

    evas=[]
    for i in range(len(population)):

        eva={"different":different(distributed_population[i],target), 
        "complexity":complexity(nor_population[i]), 
        "errors": pre_errors[i] if pre_errors[i]>=0.1 else error(nor_population[i],100),
        "gradient_error": gradient_error(distributed_population[i],target)}
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
    