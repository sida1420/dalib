import Classes as cl
import random

ob=cl.Obstacle(10,10)
ob.offset(10,10)
ob=(ob,cl.Rectangle(ob.points),None)
import Init
import Visual
import Evaluate
w=10
h=10

start=cl.Point(random.uniform(0,w*0.25),random.uniform(0,h))
while cl.pointObstalcesCollision(start,[ob]):
    start=cl.Point(random.uniform(0,w*0.25),random.uniform(0,h))

goal=cl.Point(random.uniform(w*0.75,w),random.uniform(0,h))
while cl.pointObstalcesCollision(goal,[ob]):
    goal=cl.Point(random.uniform(w*0.75,w),random.uniform(0,h))

map={"width":10,"height":10,"obstacles":[ob],"sensors":[],"start":start,"goal":goal}
path=Init.init(10,10,start,goal)
Visual.mapPath(map,path)
