import Classes as cl
import random
def mutate(map, ind, chance):
    w=map["width"]
    h=map["height"]

    sensors=map["sensors"]

    for point in ind:
        if random.random()>chance:
            continue    
        workSes=[]
        for se in sensors:
            if se[1].intersect(point):
                if se[0].intersect(point):
                    workSes.append(se[0])
        if len(workSes)==0:
            continue
        change=cl.Point(0,0)

        for se in workSes:
            change+=cl.sensorClipping(se,point)
        
        change/=len(workSes)
        point.x+=change.x
        point.y+=change.y
        

