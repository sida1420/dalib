import Evaluate
import matplotlib.pyplot as plt
import random
def distribution(population, target):
    plt.clf()

    nor_population=[Evaluate.normalize(ind) for ind in population]

    
    distributed_population=[Evaluate.probabilize(ind,1000,50) for ind in nor_population]

    avg_population=[Evaluate.averagilize(ind) for ind in distributed_population]


    for bins in avg_population:
        step=1/len(bins)
        xs=[step*(i+0.5) for i in range(len(bins))]
        ys=[height for height in bins]
        color=(random.random(),random.random(),random.random())

        plt.plot(xs, ys, 'k-', color=color)

    ys=Evaluate.averagilize([p.y for p in target])
    for i in range(len(target)):
        plt.scatter(target[i].x,ys[i])


    plt.pause(0.01)

def distribution_ind(ind):
    nor=Evaluate.normalize(ind)

    step=1/1000
    samples=[Evaluate.func(nor, step*i) for i in range(1000)]
    
    color=(random.random(),random.random(),random.random())

    plt.hist(samples, bins=50, color=color)
    plt.show()

distribution_ind([0.9160249995761678, -1.1521718404189678, 0.859520397312084, 0.09764765586903446, 0.27897878766168155])

from collections import defaultdict
import numpy as np

def fronts(inds, evas):
    if len(inds)==0:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    
    idxes = range(len(inds))
    
    x = [evas[i]["different"] for i in idxes]
    y = [evas[i]["complexity"] for i in idxes]


    
    # ADD JITTER: This allows you to see "clones" as a small cluster
    # instead of a single point.
    
    jitter_x = np.random.normal(0, 0.005 * (max(x) - min(x) + 0.1), size=len(x))
    jitter_y = np.random.normal(0, 0.005 * (max(y) - min(y) + 0.1), size=len(y))
    ax.scatter(np.array(x) + jitter_x, 
               np.array(y) + jitter_y, 
               s=30, color='red', alpha=0.7, edgecolors='none', 
               label=f'EP')
        
    ax.set_xlabel("different (Minimize)")
    ax.set_ylabel("complexity (Minimize)")
    ax.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    fig.savefig(f"EvolutionaryComputation/CustomizeProbability/Result.svg")