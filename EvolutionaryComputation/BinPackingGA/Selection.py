
def numOfBins(gen: list, bin: int):
    current=0
    count=1
    for i in gen:
        if i+current<=bin:
            current+=i
        else:
            count+=1
            current=i  
    return count   
def numOfBinsAndHowFullEach(gen: list, bin: int, maxBoxes: int):
    current=0
    point=1.0
    for i in gen:
        if i+current<=bin:
            current+=i
        else:
            point+=1.0+(bin-current)/bin/maxBoxes
            current=i  
    return point   
def evalua(gen: list, boxes, bin):
    return numOfBinsAndHowFullEach(gen,bin,len(boxes))

def selection(gens , percent, boxes , bin):
    ranking=[]
    for gen in gens:
        ranking.append((evalua(gen, boxes,bin),gen))
    ranking.sort()
    # for score, gen in ranking:
    #     print(score, gen)
    newGens=[]
    n=int(len(ranking)*percent)
    for i in range(0,n):
        newGens.append(ranking[i][1])
    return ranking[0][0],newGens