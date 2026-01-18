
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import Classes as cl
import math
from Triangulation import earClipping
import random
import Chunking
import Visual

def obstacleGen(w,h):
    ob=cl.Obstacle(w/4,h/4)
    ob.offset(w,h)
    return ob

def sensorToRect(sensor):
    le=cl.rotate(sensor.dir,sensor.angle/2)*sensor.r+sensor.center
    re=cl.rotate(sensor.dir,-sensor.angle/2)*sensor.r+sensor.center

    points=[sensor.center,le,re]
    up=sensor.center+cl.Point(0,1)*sensor.r
    down=sensor.center+cl.Point(0,-1)*sensor.r
    left=sensor.center+cl.Point(-1,0)*sensor.r
    right=sensor.center+cl.Point(1,0)*sensor.r

    if cl.orientation(le.x,le.y,sensor.center.x,sensor.center.y,up.x,up.y)<0 and cl.orientation(re.x,re.y,sensor.center.x,sensor.center.y,up.x,up.y)>0:
        points.append(up)
    if cl.orientation(le.x,le.y,sensor.center.x,sensor.center.y,down.x,down.y)<0 and cl.orientation(re.x,re.y,sensor.center.x,sensor.center.y,down.x,down.y)>0:
        points.append(down)
    if cl.orientation(le.x,le.y,sensor.center.x,sensor.center.y,left.x,left.y)<0 and cl.orientation(re.x,re.y,sensor.center.x,sensor.center.y,left.x,left.y)>0:
        points.append(left)
    if cl.orientation(le.x,le.y,sensor.center.x,sensor.center.y,right.x,right.y)<0 and cl.orientation(re.x,re.y,sensor.center.x,sensor.center.y,right.x,right.y)>0:
        points.append(right)
    return cl.Rectangle(points)
    




def gen(w, h, obstacleCount, sensorCount):


    obstacles=[]
    while len(obstacles)<obstacleCount:
        temp=obstacleGen(w,h)
        ob=[temp,cl.Rectangle(temp.points),earClipping(temp)]
        occupied=False

        for o in obstacles:
            if o[1].intersect(o[1]) and cl.triesIntersect(o[2],ob[2]):
                occupied=True
                break


        if occupied:
            continue
        obstacles.append(ob)
    sensors=[]
    chunks=Chunking.Map(w,h,w/10)

    while len(sensors)<sensorCount:
        se=cl.Sensor(w,h)
        if not cl.pointObstalcesCollision(se.center,obstacles):
            chunks.insertSensor(se.center,len(sensors))
            sensors.append((se,sensorToRect(se)))
    
    start=cl.Point(random.uniform(0,w*0.25),random.uniform(0,h))
    while cl.pointObstalcesCollision(start,obstacles):
        start=cl.Point(random.uniform(0,w*0.25),random.uniform(0,h))
    
    goal=cl.Point(random.uniform(w*0.75,w),random.uniform(0,h))
    while cl.pointObstalcesCollision(goal,obstacles):
        goal=cl.Point(random.uniform(w*0.75,w),random.uniform(0,h))

    

    return {"width":w,"height":h,"obstacles":obstacles,"sensors":sensors, "chunks":chunks,"start":start,"goal":goal}
import pickle
map=gen(100,100,20,200)

vis=Visual.MapVisualizer(map)
vis.save("map.svg")
vis.show()

with open("EvolutionaryComputation/OEMEP/map.plk","wb") as file:
    pickle.dump(map,file)
