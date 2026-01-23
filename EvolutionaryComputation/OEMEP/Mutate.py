import Classes as cl
import random
import Chunking

def sensorsPush(workSes, point):

    change=cl.Point(0,0)
    for se in workSes:
        change+=cl.sensorClipping(se,point)
    
        
    point+=change
    return point
def randomJitter(w, point):
    r=w/20
    change=cl.Point(random.uniform(-r,r),random.uniform(-r,r))
    point+=change
    return point

def mutate(map, ind, chance):
    w=map["width"]
    h=map["height"]

    sensors=map["sensors"]
    chunks=map["chunks"]

    for i in range(1,len(ind)-1):
        point=ind[i]
        if random.random()>chance:
            continue    
        idxs=chunks.getSensors(point)
        workSes=[]
        for j in idxs:
            if sensors[j][1].intersect(point):
                if sensors[j][0].intersect(point):
                    workSes.append(sensors[j][0])
        if len(workSes)==0:
            point=randomJitter(w,point)
        else:
            point=sensorsPush(workSes,point)
        if point.x<0:
            point.x=0
        elif point.x>map["width"]:
            point.x=map["width"]
        if point.y<0:
            point.y=0
        elif point.y>map["height"]:
            point.y=map["height"]
        ind[i]=point
    
        

