import random

def dominate(eva1, eva2, objectives):
    count=0
    for key in objectives:
        if eva1[key]>eva2[key]:
            return False
        if eva1[key]<eva2[key]:
            count+=1
    return count>0
import Evaluate

def updateEP(offsprings,offspringEva, externalPareto, EPEva, objectives):
    newEP=[]
    newEPEva=[]
    removed=set()

    existing_signatures = set(tuple(eva[obj] for obj in objectives) for eva in EPEva)

    for i in range(len(offsprings)):
        sig = tuple(offspringEva[i][obj] for obj in objectives)

        if sig in existing_signatures:
            continue

        nonDominated=True
        for j in range(len(externalPareto)):
            if j in removed:
                continue
            if dominate(offspringEva[i],EPEva[j],objectives):
                removed.add(j)
            elif dominate(EPEva[j],offspringEva[i],objectives):
                nonDominated=False
                break

        if nonDominated:
            newEP.append(offsprings[i])
            newEPEva.append(offspringEva[i])
            existing_signatures.add(sig)

    for j in range(len(externalPareto)):
        if j in removed:
            continue
        newEP.append(externalPareto[j])
        newEPEva.append(EPEva[j])

    return (newEP,newEPEva)


def selection(weights, population, evas, offsprings, offspringEva, neighbours, gbips, popuSize,objectives, reference, replaceLimit):
    #e, n, s
    n=len(population)

    

    for i in range(popuSize):
        localNeighs=neighbours[i]
        replaceCount=0
        for indIdx in localNeighs:

            neighGbip=gbips[indIdx]
            offspringGbip=Evaluate.gbip(weights[indIdx],offspringEva[i],reference,0.5,objectives)

            if offspringGbip<neighGbip:
                replaceCount+=1
                #replace
                population[indIdx]=offsprings[i]
                evas[indIdx]=offspringEva[i]
                gbips[indIdx]=offspringGbip

                if replaceCount==replaceLimit:
                    break



    return (population,evas,gbips)

        
        
    

    



    
    

        

    
        


    




