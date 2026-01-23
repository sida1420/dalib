import random
objectiveCount=("exposure","distance")

def dominate(eva1, eva2):
    count=0
    for key in objectiveCount:
        if eva1[key]>eva2[key]:
            return False
        if eva1[key]<eva2[key]:
            count+=1
    return count>0

def selection(inds, evaluations, popuSize):
    #e, n, s
    n=len(inds)
    storage=[[0,[]] for i in range(len(inds))]
    crowding=[0 for i in range(len(inds))]
    level=[None for i in range(n)]
    
    fronts=[[]]
    for i in range(n):
        for j in range(n):
            if i==j:
                continue
            if dominate(evaluations[j],evaluations[i]):
                storage[j][1].append(i)
                storage[i][0]+=1
        if storage[i][0]==0:
            fronts[0].append(i)
    frontIdx=0

    while frontIdx<len(fronts):
        newFront=[]

        for i in fronts[frontIdx]:
            level[i]=frontIdx
            for j in storage[i][1]:
                storage[j][0]-=1
                if storage[j][0]==0:
                    newFront.append(j)
        if len(newFront)>0:
            fronts.append(newFront)
        frontIdx+=1
    
    for front in fronts:
        for key in objectiveCount:
            temp=sorted(front,key=lambda j: evaluations[j][key])
            m= evaluations[temp[-1]][key]-evaluations[temp[0]][key]
            if m==0:
                continue
            crowding[temp[0]] = 1e9
            crowding[temp[-1]] = 1e9
            for j in range(1,len(temp)-1):
                crowding[temp[j]]+=(evaluations[temp[j+1]][key]-evaluations[temp[j-1]][key])/m
        
        front.sort(key=lambda j:crowding[j],reverse=True)
    newPopulation=[]
    newEva=[]
    newLevel=[]
    newCrowding=[]

    for front in fronts:
        if len(newPopulation)+len(front)<=popuSize:
            for i in front:
                newPopulation.append(inds[i])
                newEva.append(evaluations[i])
                newLevel.append(level[i])
                newCrowding.append(crowding[i])
        else:
            for i in front:
                if len(newPopulation)>=popuSize:
                    break
                newPopulation.append(inds[i])
                newEva.append(evaluations[i])
                newLevel.append(level[i])
                newCrowding.append(crowding[i])
            break

    return (newPopulation,newEva, newLevel, newCrowding )

    

        

    
        


    




