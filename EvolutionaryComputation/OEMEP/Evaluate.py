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
    secondNor=[]
    for i in range(len(firstNor)-1):
        secondNor.append(firstNor[i])
        obs=[]
        p1=firstNor[i]
        p2=firstNor[i+1]
        for ob in map["obstacles"]:
            if ob[1].lineIntersect(p1,p2):
                if ob[0].lineIntersect(cl.Edge(p1.x,p1.y,p2.x,p2.y)):
                    obs.append(ob[0])
        secondNor+=cl.inBetweenGen(p1,p2,obs)
    secondNor.append(firstNor[len(firstNor)-1])
    # Visual.mapPath(map,secondNor)
    return secondNor



def evaluate(map, ind):
    distance=0
    exposure=0
    # Visual.mapPath(map,ind)
    path=normalize(map,ind)

    cost=0
    for i in range(len(path)-1):
       p1=path[i]
       p2=path[i+1]
       distance+=p1.dist(p2)
    for se in map["sensors"]:
        segs=[]
        for i in range(len(path)-1):
            p1=path[i]
            p2=path[i+1]
            if se[1].lineIntersect(p1,p2):
                segs.append(cl.Edge(p1.x,p1.y,p2.x,p2.y))
        obs=[]
        for ob in map["obstacles"]:
            if se[1].intersect(ob[1]):
                obs.append(ob[0])
        for c in se[0].casting(obs,segs):
            cost+=c
    # print(cost)
    # print(distance)
    # Visual.mapPath(map,path)
    return {"exposure":cost,"distance":distance,"simplixity": len(path)}










