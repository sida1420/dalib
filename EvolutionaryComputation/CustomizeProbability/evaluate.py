import random

def func(ind, x):
    ans=0
    for i in len(ind):
        ans+=ind[i]*(x**(i+1))
    return ans

def normalize(ind):
    y1=func(ind,1)

    

def probabilize(ind, sample_size, num_bins):
    samples=[func(ind, random.random()) for i in range(sample_size)]

    bins=[0]*num_bins
    wrong=0
    step=1/num_bins
    for sample in samples:
        if sample<0 or sample>1:
            wrong+=1
        else:
            bins[sample//step]+=1

    s=sum(bins)
    for bin in bins:
        bin/=s

    return bins





def diffrent(ind, target):
    n=len(target)

    