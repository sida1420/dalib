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
    def __truediv__(self,other):
        return Point(self.x/other,self.y/other)

    def __eq__(self,other):
        return self.x==other.x and self.y==other.y
    def dist(self,other):
        return math.sqrt((self.x-other.x)**2+(self.y-other.y)**2)

    def __abs__(self):
        return math.sqrt(self.x**2+self.y**2)

    def __repr__(self):
        return f"{round(self.x)} {round(self.y)}"
    def __call__(self):
        return (self.x,self.y)
    def __hash__(self):
        return hash((self.x, self.y))
    


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

def angle(vec1, vec2):
    return math.acos((vec1*vec2)/(abs(vec1)*abs(vec2)))


def intersection(p, e):
    p1=Point(e.x1,e.y1)
    p2=Point(e.x2,e.y2)

    if p2.y==p1.y:
        return None
    
    t=(p.y-p1.y)/(p2.y-p1.y)
    

    if t<0 or t>1:
        return None
    return Point(t*(p2.x-p1.x)+p1.x,p.y)


class Obstacle:
    def __init__(self, w=None, h=None, points=None, coeff=None):
        self.coeff=random.random() if coeff is None else coeff
        if points is not None:
            self.points=points
            return
        w=w/2
        h=h/2
        self.points=[Point(rd.uniform(-w,w),rd.uniform(-h,h)) for i in range(4)]
        origin=Point(0,0)
        for p in self.points:
            origin+=p
        origin/=len(self.points)

        a_p=[(toAngle(point-origin),point) for point in self.points]
        a_p.sort(key=lambda data: data[0])
        self.points=[data[1] for data in a_p]


        edges=set()

        for i in range(4):
            edge=(self.points[i],self.points[(i+1)%4])

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


                midpoint=(p2-p1)*rd.uniform(0,1)+p1
                direction=(p2-p1)/p2.dist(p1)
                direction=Point(direction.y,-direction.x)*rd.choice([1,-1])
                distance=rd.uniform(0,abs(p2-p1)*0.5)
                attempt=midpoint+direction*distance

                i=0
                while i<5 and ((attempt.x<-w or attempt.y<-h or attempt.x>w or attempt.y >h) or (collide(Edge(attempt.x,attempt.y,p1.x,p1.y)) or collide(Edge(attempt.x,attempt.y,p2.x,p2.y)))):
                    distance/=2
                    attempt=midpoint+direction*distance
                    i+=1
                if i<5:

                    edges.remove(edge)

                    midpoint=attempt
                
                    edges.add((p1,midpoint))
                    edges.add((midpoint,p2))
                
                    ans+=reccusion(p1, midpoint,chance/2)+[midpoint]+reccusion(midpoint, p2, chance/2)

            return ans
        
        newPoints=[]
        for i in range(4):
            newPoints.append(self.points[i])
            newPoints+=reccusion(self.points[i],self.points[(i+1)%4],0.3)
        
        self.points=newPoints



    def offset(self,w,h):
        offX=rd.uniform(0,w)
        offY=rd.uniform(0,h)

        for p in self.points:
            p.x+=offX
            p.y+=offY
    def pointIntersect(self, p):
        left=0
        right=0
        for i in range(len(self.points)):
            edge=Edge(self.points[i].x,self.points[i].y,self.points[(i+1)%len(self.points)].x,self.points[(i+1)%len(self.points)].y)
            inter=intersection(p,edge)
            if inter is None:
                continue
            if inter.x>p.x:
                right+=1
            else:
                left+=1
        return right%2==1 and left%2==1

    def lineIntersect(self, e):
        for i in range(len(self.points)):
            edge=Edge(self.points[i].x,self.points[i].y,self.points[(i+1)%len(self.points)].x,self.points[(i+1)%len(self.points)].y)
            if e.intersect(edge):
                return True
        return False


    def draw(self):
        return [(p.x, p.y) for p in self.points]

class Triangle(Obstacle):
    def pointIntersect(self, p):
        d1=orientation(self.points[0].x,self.points[0].y,self.points[1].x,self.points[1].y,p.x,p.y)
        d2=orientation(self.points[1].x,self.points[1].y,self.points[2].x,self.points[2].y,p.x,p.y)
        d3=orientation(self.points[2].x,self.points[2].y,self.points[0].x,self.points[0].y,p.x,p.y)

        return d1<0 and d2<0 and d3<0

class Sensor:
    def __init__(self, w, h, r=None, a=None):

        self.center=Point(random.uniform(0,w),random.uniform(0,h))

        
        self.r=r if r is not None else  random.uniform(h/20,h/10)
        self.angle=a if a is not None else random.uniform(math.pi/6,math.pi*2)
        self.dir=toDir(random.uniform(-math.pi,math.pi))

    def thetaL(self):
        return toAngle(rotate(self.dir,-self.angle/2)) *180/math.pi     
    def thetaR(self):
        return toAngle(rotate(self.dir,self.angle/2))*180/math.pi
        
class Rectangle:
    def __init__(self, points):
        self.x1=0
        self.y1=0
        self.x2=0
        self.y2=0
        for p in points:
            self.x1=min(self.x1,p.x)
            self.y1=min(self.y1,p.y)
            self.x2=max(self.x2,p.x)
            self.y2=max(self.y2,p.y)

    def intersect(self, other):
        if isinstance(other,Rectangle):
            return (self.x1<other.x2 and self.x2>other.x1 and self.y1<other.y2 and self.y2>other.y1)
        return other.x>self.x1 and other.y>self.y1 and other.x<self.x2 and other.y<self.y2