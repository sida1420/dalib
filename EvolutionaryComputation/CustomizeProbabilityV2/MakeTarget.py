from Classes import Point
def clamp(points):
    miX=points[0].x
    maX=points[0].x
    for p in points:
        miX=min(miX,p.x)
        maX=max(maX,p.x)

    new_points=[Point((p.x-miX)/(maX-miX),p.y) for p in points]
    return new_points

def inbetween(p1, p2, desired_length):
    if p2.x-p1.x<=desired_length:
        return []
    
    mp=(p1+p2)/2
    return inbetween(p1, mp,desired_length*2)+[mp]+inbetween(mp,p2,desired_length*2)

def connect(points):
    new_points=[points[0],]
    for i in range(1,len(points)):
        new_points+=inbetween(points[i-1],points[i],0.1)+[points[i]]
    return new_points