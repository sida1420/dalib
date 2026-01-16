import matplotlib.pyplot as plt
import matplotlib.patches as patches
def drawObstacle(obstacles, fig, ax):

    for o in obstacles:
        polygon = patches.Polygon(o[0].draw(), 
                            closed=True, facecolor='darkgrey', 
                                  edgecolor='black',
                            linewidth=1,
                            alpha=0.8)
        
        # 3. Add to the axis
        ax.add_patch(polygon)

def drawMap(map,fig, ax):
    drawObstacle(map["obstacles"],fig,ax)
    for s in map["sensors"]:
        wedge=patches.Wedge(s[0].center(),s[0].r,s[0].thetaL(),s[0].thetaR(),alpha=0.5)
        plt.scatter(s[0].center.x, s[0].center.y, s=1)
        ax.add_patch(wedge)
    
    ax.autoscale()
    ax.set_aspect('equal')
    plt.scatter(map["start"].x, map["start"].y, s=50)
    plt.scatter(map["goal"].x, map["goal"].y, s=50)

import random

def drawPath(path):
    color=(random.random(),random.random(),random.random())
    xs=[]
    ys=[]
    for i in range(len(path)):
        xs.append(path[i].x)
        ys.append(path[i].y)
        plt.scatter(path[i].x, path[i].y, s=10,color=color)
    plt.plot(xs,ys,linewidth=2,color=color)


def map(map):
    fig, ax = plt.subplots()
    drawMap(map,fig,ax)

    plt.savefig("EvolutionaryComputation/OEMEP/map.jpeg")
    plt.show()
import Evaluate

def mapPath(map,path):
    fig, ax = plt.subplots()
    drawMap(map,fig,ax)
    drawPath(Evaluate.normalize(map,path))

    plt.show()

def mapPaths(map,paths):
    fig, ax = plt.subplots()
    drawMap(map,fig,ax)
    for path in paths:
        drawPath(Evaluate.normalize(map,path))

    plt.show()