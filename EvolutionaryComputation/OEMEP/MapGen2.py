
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import Classes as cl
import math
from Triangulation import earClipping




def obstacleGen(w,h):
    ob=cl.Obstacle(w/2,h/2)
    ob.offset(w,h)
    return earClipping(ob)


def gen(w, h, obstacleCount, sensorCount):


    obstacles=[]
    for i in range(obstacleCount):
        obstacles+=obstacleGen(w,h)
    sensors=[]
    for i in range(sensorCount):
        sensors.append(cl.Sensor(w,h))

    fig, ax = plt.subplots()
    for o in obstacles:
        polygon = patches.Polygon(o.draw(), 
                            closed=True, facecolor='darkgrey', 
                                  edgecolor='black',
                            linewidth=1,
                            alpha=0.8) # alpha makes it slightly transparent
        
        # 3. Add to the axis
        ax.add_patch(polygon)
    for s in sensors:
        wedge=patches.Wedge(s.center(),s.r,s.thetaL(),s.thetaR(),alpha=0.5)
        plt.scatter(s.center.x, s.center.y, s=1)
        ax.add_patch(wedge)
        
    ax.autoscale()
    ax.set_aspect('equal')
    
    plt.show()
gen(100,100,5,50)