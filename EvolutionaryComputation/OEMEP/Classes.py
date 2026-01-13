import math
import random
class Point:
    def __init__(self,x,y):
        self.x=x
        self.y=y

    def __add__(self,other):
        return Point(self.x+other.x,self.y+other.y)
    def __sub__(self,other):
        return Point(self.x-other.x,self.y-other.y)
    #scale
    def __mul__(self,other):
        if isinstance(other,Point):
            return self.x*other.x+ self.y*other.y
        return Point(self.x*other,self.y*other)

    def dist(self,other):
        return math.sqrt((self.x-other.x)**2+(self.y-other.y)**2)

    def __abs__(self):
        return math.sqrt(self.x**2+self.y**2)

    def __repr__(self):
        return f"{round(self.x)} {round(self.y)}"


def orientation(x1, y1, x2, y2, x3, y3):
    return (y2-y1)*(x3-x2)-(x2-x1)*(y3-y2)

class Edge:
    def __init__(self,x1,y1,x2,y2):
        self.x1=x1
        self.x2=x2
        self.y1=y1
        self.y2=y2


    def intersect(self,other):
        #
        abc=orientation(self.x1,self.y1,self.x2,self.y2,other.x1,other.y1)
        abd=orientation(self.x1,self.y1,self.x2,self.y2,other.x2,other.y2)
        cda=orientation(other.x1,other.y1,other.x2,other.y2,self.x1,self.y1)
        cdb=orientation(other.x1,other.y1,other.x2,other.y2,self.x2,self.y2)

        return abc*abd<0 and cda*cdb<0



import random as rd

def toAngle(dir):
    d=abs(dir)
    return math.atan2(dir.y/d,dir.x/d)

def toDir(theta):
    """
    theta in range -PI to PI
    """
    
    theta*=math.pi
    return Point(math.cos(theta),math.sin(theta))

def rotate(dir, angle):
    theta=toAngle(dir)
    theta+=angle
    return Point(math.cos(theta),math.sin(theta))

class Obstacle:
    def __init__(self, w, h):
        w=w/2
        h=h/2
        self.points=[Point(rd.uniform(-w,w),rd.uniform(-h,h)) for i in range(3)]

        a_p=[(toAngle(point),point) for point in self.points]
        a_p.sort(key=lambda data: data[0])
        self.points=[data[1] for data in a_p]


        edges=set()

        for i in range(3):
            edge=(self.points[i],self.points[(i+1)%3])

            edges.add(edge)


        def collide(e):
            for te in edges:
                if Edge(te[0].x,te[0].y,te[1].x,te[1].y).intersect(e):
                    return True

            return False


        def reccusion(p1,p2, chance):

            ans=[]

            if rd.random()<chance:
                edge=(p1,p2)


                midpoint=(p2-p1)*rd.random()+p1
                direction=Point(rd.uniform(-math.pi,math.pi),rd.uniform(-math.pi,math.pi))
                distance=rd.uniform(0,abs(p2-p1)/2)
                attempt=midpoint+direction*distance

                i=0
                while i<10 and (collide(Edge(attempt.x,attempt.y,p1.x,p1.y)) or collide(Edge(attempt.x,attempt.y,p2.x,p2.y))):
                    distance/=2
                    attempt=midpoint+direction*distance
                    i+=1
                if i<10:

                    edges.remove(edge)

                    midpoint=attempt
                
                    edges.add((p1,midpoint))
                    edges.add((midpoint,p2))
                
                    ans+=reccusion(p1, midpoint,chance/2)+[midpoint]+reccusion(midpoint, p2, chance/2)

            return ans
        
        newPoints=[]
        for i in range(3):
            newPoints.append(self.points[i])
            newPoints+=reccusion(self.points[i],self.points[(i+1)%3],0.5)
        
        self.points=newPoints


        self.coeff=random.random()

    def offset(self,w,h):
        offX=rd.uniform(0,w)
        offY=rd.uniform(0,h)

        for p in self.points:
            p.x+=offX
            p.y+=offY


        
        
