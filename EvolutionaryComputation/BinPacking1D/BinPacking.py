import random
from collections import defaultdict
import queue
import matplotlib
matplotlib.use("TkAgg")   # or "Agg" (no GUI)

import matplotlib.pyplot as plt

import Initialize,Selection,Crossover,Mutation, StandardDeviation

def binPackingGA(boxes:list, bin: int):

    #How many in each generation
    gensSize=100
    #Loop
    end=1000
    bestScore=0
    bestGen=[]
    
    gens=[Initialize.init(boxes) for i in range(0,gensSize)]

    #Selection percent: 35% elites - 5% diversities
    #Crossover percent: 100%
    #Mutation percent: 40% (until meet gen size)

    #Graph
    avers=[]
    iterations=[]
    sds=[]

    for i in range(0,end):
        print("\tNEW GEN")

        fitnesses=[Selection.evalua(gen,boxes,bin) for gen in gens]
        avers.append(StandardDeviation.Average(fitnesses))
        iterations.append(i)
        sds.append(StandardDeviation.StandardDeviation(fitnesses))

        score, gens=Selection.selection(gens,0.35,0.05,boxes,bin)
        if bestScore<score:
            bestScore=score
            bestGen=gens[0]

        gens=Crossover.crossover(gens,1)
        gens=Mutation.mutation(gens,0.4, gensSize).copy()

    print("\tBEST\n",bestScore,Selection.numOfBins(bestGen,boxes,bin),[boxes[i] for i in bestGen])

    # plt.plot(iterations,avers,label="Average")
    plt.plot(iterations,sds,label="StandardDeviation")
    plt.legend()
    plt.xlabel('Generation')
    plt.show()


    
    return bestScore
    

def binPackingBF(boxes: list, bin: int):
    
    best=[len(boxes)]

    def dfs(current, binUsed, boxUsed, left,best):

        if binUsed>best[0]:
            return
        if left==0:
            best[0]=min(best[0],binUsed)
            return

        for b in boxes:
            if boxUsed[b]==0:
                continue
            
            boxUsed[b]-=1
            if b+current>bin:
                dfs(b, binUsed+1,boxUsed,left-1,best)
            else:
                dfs(current+b,binUsed,boxUsed,left-1,best)

                
            boxUsed[b]+=1
    
        
    boxUsed={}
    for box in boxes:
        boxUsed[box]=boxUsed.get(box,0)+1

    dfs(0,1,boxUsed,len(boxes),best)

    return best[0]

    
def read_falkenauer(path):
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]

    idx = 0
    P = int(lines[idx]); idx += 1
    problems = []

    for _ in range(P):
        pid = lines[idx]; idx += 1
        cap, n, best = map(int, lines[idx].split())
        idx += 1

        items = list(map(int, lines[idx:idx+n]))
        idx += n

        problems.append({
            "id": pid,
            "capacity": cap,
            "items": items,
            "best": best
        })

    return problems





problems=read_falkenauer("/home/daxanity/Code/python/dalib/EvolutionaryComputation/BinPacking1D/dataset.txt")

items=problems[2]["items"]
bin_capacity=problems[2]["capacity"]

binPackingGA(items,bin_capacity)
print("Actual best = ",problems[2]["best"])


