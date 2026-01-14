
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import Classes as cl
import math
from Triangulation import earClipping
import random



def obstacleGen(w,h):
    ob=cl.Obstacle(w/4,h/4)
    ob.offset(w,h)
    return ob



def gen(w, h, obstacleCount, sensorCount):


    obstacles=[]
    while len(obstacles)<obstacleCount:
        ob=obstacleGen(w,h)

        obstacles.append([ob,cl.Rectangle(ob.points),earClipping(ob)])
    sensors=[]

    while len(sensors)<sensorCount:
        se=cl.Sensor(w,h)
        occupied=False
        for o in obstacles:
            if o[1].intersect(se.center):
                if o[0].pointIntersect(se.center):
                    occupied=True
                    break
        if not occupied:
            sensors.append(se)
    
    start=cl.Point(random.uniform(0,w*0.25),random.uniform(0,h))
    def occupied(p):
        for o in obstacles:
            if o[1].intersect(p):
                if o[0].pointIntersect(p):
                    return True
        return False
    while occupied(start):
        start=cl.Point(random.uniform(0,w*0.25),random.uniform(0,h))
    
    goal=cl.Point(random.uniform(w*0.75,w),random.uniform(0,h))
    while occupied(goal):
        goal=cl.Point(random.uniform(w*0.75,w),random.uniform(0,h))

    fig, ax = plt.subplots()
    for o in obstacles:
        polygon = patches.Polygon(o[0].draw(), 
                            closed=True, facecolor='darkgrey', 
                                  edgecolor='black',
                            linewidth=1,
                            alpha=0.8)
        
        # 3. Add to the axis
        ax.add_patch(polygon)
    for s in sensors:
        wedge=patches.Wedge(s.center(),s.r,s.thetaL(),s.thetaR(),alpha=0.5)
        plt.scatter(s.center.x, s.center.y, s=1)
        ax.add_patch(wedge)
    
    plt.scatter(start.x, start.y, s=50)
    plt.scatter(goal.x, goal.y, s=50)

    ax.autoscale()
    ax.set_aspect('equal')
    
    plt.savefig("EvolutionaryComputation/OEMEP/map.jpeg")
    plt.show()

    return {"width":w,"height":h,"obstacles":obstacles,"sensors":sensors,"start":start,"goal":goal}
import pickle

with open("EvolutionaryComputation/OEMEP/map.plk","wb") as file:
    pickle.dump(gen(100,100,10,100),file)
