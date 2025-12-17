# TO DO: change for index evaluation
def numOfBins(gen: list, bin: int, boxes):
    current=0
    count=1
    for i in gen:
        if boxes[i]+current<=bin:
            current+=boxes[i]
        else:
            count+=1
            current=boxes[i]  
    return count   


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

def sqrtHowFullEach(gen: list, bin:  int, boxes):
    point=0
    current=0
    count=1
    for i in gen:
        if boxes[i]+current<=bin:
            current+=boxes[i]
        else:
            count+=1
            point+=(current)**0.5
            current=boxes[i]
    return point 
#

def evalua(gen: list, boxes, bin):
    return sqrtHowFullEach(gen,bin,boxes)


def selection(gens , percent, boxes , bin):
    ranking=[]
    for gen in gens:
        ranking.append((evalua(gen, boxes,bin),gen))
    ranking.sort()

    i=3
    for score, gen in ranking:
        rep=[boxes[i] for i in gen]
        num=numOfBins(gen,bin,boxes)
        print(score,num, rep)
        i-=1
        if not i:
            break

    newGens=[]
    n=int(len(ranking)*percent)
    for i in range(0,n):
        newGens.append(ranking[i][1])
    return ranking[0][0],newGens