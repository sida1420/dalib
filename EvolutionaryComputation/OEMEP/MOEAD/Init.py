import Classes as cl
from Vector import Vector
import random
import math
def init(w,h,start, goal):
    r=w/10

    points=[start]

    current=cl.Point(0,random.uniform(0,h))

    points.append(current)
    halfPI=math.pi/2
    while True:
        dir=cl.toDir(random.uniform(-halfPI,halfPI))
        attempt=current+dir*r
        if attempt.x<0 or attempt.y<0 or attempt.y>h:
            continue

        if attempt.x>w:
            break
        current=attempt
        points.append(attempt)
    return points+[goal]


def initWeightVectors(numDivisions, numObjectives):

    ans=[]

    step=1/numDivisions
    divisions=[i*step for i in range(numDivisions+1)]

    def recursion(i, sum, task):
        if i==numObjectives:
            if abs(sum-1)<=1e-9:
                ans.append(Vector(task.copy()))
            
            return
        for j in divisions:
            if sum+j>1+1e-9:
                return
            task.append(j)
            recursion(i+1,sum+j,task)
            task.pop()

    recursion(0,0,[])
    return ans

def whoAmINeighbour(current,weights, numNeighbours):
    neighs=[]

    dists=[]
    for idx in range(len(weights)):
        dists.append((weights[current].dist(weights[idx]),idx))
    dists.sort()

    ans=[dists[i][1] for i in range(numNeighbours)]
    return ans


def initReferencePoint(evas, objectives):
    ans=[float('inf') for obj in objectives]

    for eva in evas:
        for i in range(len(objectives)):
            ans[i]=min(ans[i],eva[objectives[i]])
    return Vector(ans)
