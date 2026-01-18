import Classes as cl
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


