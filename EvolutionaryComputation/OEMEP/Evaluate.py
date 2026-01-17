import Classes as cl
import matplotlib.pyplot as plt
import Visual

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
    while not nothing:
        print(iter)
        nothing=True
        newNor=[]
        for i in range(len(secondNor)-1):
            newNor.append(secondNor[i])
            p1=secondNor[i]
            p2=secondNor[i+1]
            if (p1,p2) in save:
                continue
            obs=[]

            for ob in map["obstacles"]:
                if ob[1].lineIntersect(p1,p2):
                    if ob[0].lineIntersect(cl.Edge(p1.x,p1.y,p2.x,p2.y)):
                        obs.append(ob[0])
            if len(obs)==0:
                save.add((p1,p2))
            else:
                mid=cl.inBetweenGen(p1,p2,obs,0)
                if not mid:
                    save.add((p1,p2))
                else:
                    newNor+=mid
                    nothing=False
        newNor.append(secondNor[len(secondNor)-1])
        secondNor=newNor
        iter+=1
    # Visual.mapPath(map,secondNor)
    print(iter)
    return secondNor



def evaluate(map, inds):
    evas=[]
    paths=[]
    for ind in inds:
        # Visual.mapPath(map,ind)
        path=normalize(map,ind)

        distance=0
        for i in range(len(path)-1):
           p1=path[i]
           p2=path[i+1]
           distance+=p1.dist(p2)
        evas.append({"exposure":0,"distance":distance,"complexity":len(path)})
        paths.append(path)
    for se in map["sensors"]:
        segs=[]

        for i in range(len(paths)):
            path=paths[i]
            for j in range(len(path)-1):    
                p1=path[j]
                p2=path[j+1]
                if se[1].lineIntersect(p1,p2):
                    segs.append((i,j,cl.Edge(p1.x,p1.y,p2.x,p2.y)))
        obs=[]
        for ob in map["obstacles"]:
            if se[1].intersect(ob[1]):
                obs.append(ob[0])
        costs=se[0].casting(obs,segs)
        for i in range(len(segs)):
            evas[segs[i][0]]["exposure"]+=costs[i]
    # print(cost)
    # print(distance)
    # Visual.mapPath(map,path)
    return evas










