import Classes as cl
import matplotlib.pyplot as plt
import Visual
import multiprocessing as mp
from functools import partial

def normalize(map, ind):
    firstNor=[]
    if len(ind)>0:
        firstNor.append(ind[0])

    for i in range(1,len(ind)-1):
        if not cl.pointObstalcesCollision(ind[i],map["obstacles"]):
            firstNor.append(ind[i])
    
    if len(ind)>1:
        firstNor.append(ind[len(ind)-1])

    # Visual.mapPath(map,firstNor)
    secondNor=firstNor
    save=set()
    nothing=False
    iter=0
    max_iter=5
    while not nothing and iter<max_iter:
        # print(iter)
        # print(save,"-", secondNor)
        nothing=True
        newNor=[]
        force=2**(-iter)
        for i in range(len(secondNor)-1):
            newNor.append(secondNor[i])
            p1=secondNor[i]
            p2=secondNor[i+1]
            if (p1,p2) in save:
                continue
            obs=[]

            for ob in map["obstacles"]:
                if ob[1].lineIntersect(p1,p2):
                    if ob[0].lineIntersect(p1,p2):
                        obs.append(ob[0])
            if len(obs)==0:
                save.add((p1,p2))
            else:
                mid=cl.inBetweenGen(p1,p2,obs,force)
                if not mid:
                    save.add((p1,p2))
                else:
                    newNor+=mid
                    nothing=False
        newNor.append(secondNor[-1])
        secondNor=newNor
        iter+=1
    # Visual.mapPath(map,secondNor)
    # print(iter)
    return secondNor



def evaluate(map, inds):
    evas=[]
    paths=[]
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # partial lets us pass the 'map' argument to every call
        func = partial(normalize,map)
        paths = pool.map(func, inds)
    for path in paths:
        distance=0
        for i in range(len(path)-1):
           distance+=path[i].dist(path[i+1])
        evas.append({"exposure":0,"distance":distance,"complexity":len(path)})
    for se in map["sensors"]:
        segs=[]

        for i in range(len(paths)):
            path=paths[i]
            for j in range(len(path)-1):    
                p1=path[j]
                p2=path[j+1]
                if se[1].lineIntersect(p1,p2):
                    segs.append((i,j,p1,p2))
        obs=[]
        for ob in map["obstacles"]:
            if se[1].intersect(ob[1]):
                obs.append(ob[0])
        costs=se[0].casting(obs,segs)
        for i in range(len(segs)):
            evas[segs[i][0]]["exposure"]+=costs[i]
    return evas










