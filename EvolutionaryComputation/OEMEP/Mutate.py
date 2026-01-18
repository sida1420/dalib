import Classes as cl
import random
import Chunking
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
        for i in idxs:
            if sensors[i][1].intersect(point):
                if sensors[i][0].intersect(point):
                    workSes.append(sensors[i][0])
        if len(workSes)==0:
            continue
        change=cl.Point(0,0)

        for se in workSes:
            change+=cl.sensorClipping(se,point)
        
        # change/=len(workSes)

            

        point.x+=change.x
        point.y+=change.y
        if point.x<0:
            point.x=0
        elif point.x>map["width"]:
            point.x=map["width"]
        if point.y<0:
            point.y=0
        elif point.y>map["height"]:
            point.y=map["height"]
        

