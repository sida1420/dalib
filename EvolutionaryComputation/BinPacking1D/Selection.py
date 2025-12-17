

def firstFit(gen, boxes, bin):
    bins=[]

    for i in gen:
        
        needMore=True
        for j in range(len(bins)):
            if bins[j]+boxes[i]<=bin:
                bins[j]+=boxes[i]
                needMore=False
                break
        if needMore:
            bins.append(boxes[i])
    return bins

def numOfBins(gen, boxes, bin):
    return len(firstFit(gen,boxes,bin))

#error last bin
# def numOfBinsNextFit(gen: list, bin: int, boxes):
#     current=0
#     count=1
#     for i in gen:
#         if boxes[i]+current<=bin:
#             current+=boxes[i]
#         else:
#             count+=1
#             current=boxes[i]  
#     return count   

# def numOfBinsAndHowFullEach(gen: list, bin: int, maxBoxes: int):
#     current=0
#     point=1.0
#     for i in gen:
#         if i+current<=bin:
#             current+=i
#         else:
#             point+=1.0+(bin-current)/bin/maxBoxes
#             current=i  
#     return point   

def powHowFullEach(gen: list, boxes, bin):
    point=0
    bins=firstFit(gen,boxes,bin)
    for b in bins:
        point+=(b/bin)**2
    return point 


def evalua(gen: list, boxes, bin):
    return powHowFullEach(gen,boxes,bin)


def selection(gens , elitePercent, diversePercent, boxes , bin):
    ranking=[]
    for gen in gens:
        ranking.append((evalua(gen, boxes,bin),gen))

    ranking.sort(key=lambda x: x[0], reverse=True)


    c=3
    for score, gen in ranking:
        rep=[boxes[i] for i in gen]
        num=numOfBins(gen,boxes,bin)
        print(score,num, rep)
        c-=1
        if not c:
            break

    newGens=[]
    n=int(len(ranking)*elitePercent)
    diverseBins=set()

    for i in range(0,n):
        newGens.append(ranking[i][1])
        diverseBins.add(numOfBins(ranking[i][1],boxes,bin))

    m=int(len(ranking)*diversePercent)


    for i in range(n,len(ranking)):
        if m<=0:
            break
        tempBin=numOfBins(ranking[i][1],boxes,bin)
        if tempBin not in diverseBins:
            m-=1
            diverseBins.add(tempBin)
            newGens.append(ranking[i][1])





    
    return ranking[0][0],newGens