import math
import random
class Point:
    __slots__ = ['x', 'y']
    def __getstate__(self):
        return (self.x, self.y)
    def __setstate__(self, state):
        self.x, self.y = state
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
    def __neg__(self):
        return Point(-self.x,-self.y)
    def __eq__(self,other):
        return self.x==other.x and self.y==other.y
    def dist(self,other):
        return math.sqrt((self.x-other.x)**2+(self.y-other.y)**2)
    def cross(self, other):
        return self.x*other.y-self.y*other.x

    def __abs__(self):
        return math.sqrt(self.x**2+self.y**2)

    def __repr__(self):
        return f"{round(self.x,1)} {round(self.y,1)}"
    def __call__(self):
        return (self.x,self.y)
    def __hash__(self):
        return hash((self.x, self.y))
    


def orientation(x1, y1, x2, y2, x3, y3):
    return (y2-y1)*(x3-x2)-(x2-x1)*(y3-y2)

def onSegment(x1,y1,x2,y2,x3,y3):
    if min(x1,x2)<=x3 <=max(x1,x2) and min(y1,y2)<=y3<=max(y1,y2):
        return True
    return False

class Edge:
    __slots__ = ['x1', 'y1', 'x2', 'y2']
    def __getstate__(self):
        return (self.x1, self.y1, self.x2, self.y2)
    def __setstate__(self, state):
        self.x1, self.y1, self.x2,self.y2 = state
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


        # if abs(abc) < 1e-9 and onSegment(self.x1,self.y1,self.x2,self.y2,other.x1,other.y1): return True
        # if abs(abd) < 1e-9 and onSegment(self.x1,self.y1,self.x2,self.y2,other.x2,other.y2): return True
        # if abs(cda) < 1e-9 and onSegment(other.x1,other.y1,other.x2,other.y2,self.x1,self.y1): return True
        # if abs(cdb) < 1e-9 and onSegment(other.x1,other.y1,other.x2,other.y2,self.x2,self.y2): return True


        return abc*abd<0 and cda*cdb<0
    def rayIntersect(self, p1, p2):
        # Extract coordinates to avoid object creation overhead
        p1x, p1y = p1.x, p1.y
        # p2 is the end of the ray, p1 is the origin
        
        # Vector r = p2 - p1
        rx = p2.x - p1x
        ry = p2.y - p1y
        
        # Vector s = (x2-x1, y2-y1)
        sx = self.x2 - self.x1
        sy = self.y2 - self.y1

        # Cross product of r and s
        down = rx * sy - ry * sx
        
        # Parallel lines check
        if abs(down) < 1e-9:
            return None

        # Vector (self.p1 - p1)
        dx = self.x1 - p1x
        dy = self.y1 - p1y

        # t = (q - p) x s / (r x s)
        t = (dx * sy - dy * sx) / down
        
        # u = (q - p) x r / (r x s)
        u = (dx * ry - dy * rx) / down

        # Standard intersection checks
        if t <= 0 or t >= 1:
            return None
        if u < 0 or u > 1:
            return None

        return t
    def same(self,other):
        return abs(self.x1-other.x1)<1e-9 and abs(self.x2-other.x2)<1e-9 and abs(self.y1 - other.y1)<1e-9 and abs(self.y2-other.y2)<1e-9

import random as rd

def toAngle(dir):
    d=abs(dir)
    return math.atan2(dir.y/d,dir.x/d)

def toDir(theta):
    """
    theta in range -PI to PI
    """
    
    return Point(math.cos(theta),math.sin(theta))

def rotate(dir, angle):
    theta=toAngle(dir)
    theta+=angle
    return Point(math.cos(theta),math.sin(theta))

def angle(vec1, vec2):
    
    x = (vec1 * vec2) / (abs(vec1) * abs(vec2))
    return math.acos(max(-1.0, min(1.0, x)))


def intersection(p, e):
    p1=Point(e.x1,e.y1)
    p2=Point(e.x2,e.y2)

    if abs(p2.y-p1.y)<1e-9:
        return None
    
    t=(p.y-p1.y)/(p2.y-p1.y)
    if t<0 or t>1:
        return None

    return Point(p1.x+t*(p2.x-p1.x),p1.y)

import Helper

class Obstacle:
    def __init__(self, w=None, h=None, points=None, coeff=None, sorted=True):
        if points is not None:
            self.points=points
        else:
            w=w/2
            h=h/2
            self.points=[Point(rd.uniform(-w,w),rd.uniform(-h,h)) for i in range(4)]
        self.origin=Point(0,0)

        for p in self.points:
            self.origin+=p
            
        self.origin/=len(self.points)
        self.coeff=random.random() if coeff is None else coeff
        if w is not None:
            self.coeff/=w

        if points is None or not sorted:
            a_p=[(toAngle(point-self.origin),point) for point in self.points]
            a_p.sort(key=lambda data: data[0])
            self.points=[data[1] for data in a_p]
        if points is not None:
            return


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

    def lineIntersect(self, p1, p2):
        if self.pointIntersect(p1) or self.pointIntersect(p2):
            return True

        n = len(self.points)
        for i in range(n):
            a = self.points[i]
            b = self.points[(i+1)%n]
            if Helper.fast_lines_intersect(p1.x, p1.y, p2.x, p2.y, a.x, a.y, b.x, b.y):
                return True
        return False

    def rayIntersect(self, p1, p2):
        ts = []
        n = len(self.points)
        
        # Pre-calculate Ray Vector (optimization)
        rx = p2.x - p1.x
        ry = p2.y - p1.y
        p1x, p1y = p1.x, p1.y

        for i in range(n):
            a = self.points[i]
            b = self.points[(i+1) % n]
            
            sx = b.x - a.x
            sy = b.y - a.y
            
            down = rx * sy - ry * sx
            if abs(down) < 1e-9:
                continue

            dx = a.x - p1x
            dy = a.y - p1y

            t = (dx * sy - dy * sx) / down
            u = (dx * ry - dy * rx) / down

            if 0 < t < 1 and 0 <= u <= 1:
                ts.append(t)
        ts.sort()
        ans=[]
        if len(ts)>0:
            ans.append((ts[0],0))
        for i in range(1,len(ts)):
            if i%2==0:
                ans.append((ts[i],ans[i-1][1]))
            else:
                ans.append((ts[i],ans[i-1][1]+(ts[i]-ts[i-1])))
        
        return ans
        

    def draw(self):
        return [(p.x, p.y) for p in self.points]


def spread(obstacle, force):
    ans=[]
    n=len(obstacle.points)
    for i in range(n):
        p=obstacle.points[i]
        a=obstacle.points[(i-1)%n]
        b=obstacle.points[(i+1)%n]
        pa=a-p
        pb=b-p
        pa/=abs(pa)
        pb/=abs(pb)
        bisector=pa+pb
        if orientation(a.x,a.y,p.x,p.y,b.x,b.y)<0:
            bisector=-bisector


        ans.append(p+bisector*force)
    return ans


import bisect
def cost(p1, p2, obstacles, preSegTs):
    tss=[]
    #list of list for each obstacle's ts

    for ob in obstacles:
        ts=ob.rayIntersect(p1,p2)
        tss.append(ts)
    
    ans=[]
    for j in range(len(preSegTs)):
        total=0
        t=(preSegTs[j],0.0)
        if t[0] is None:
            ans.append(0)
            continue
        for i in range(len(tss)):
            if tss[i] is None:
                continue
            idx=bisect.bisect_right(tss[i],t)
            if idx<=len(tss[i]):
                if idx==0:
                    continue
                idx-=1
                if idx%2==0:
                    total+=(t[0]-tss[i][idx][0]+tss[i][idx][1])*obstacles[i].coeff*p1.dist(p2)
                else:
                    total+=tss[i][idx][1]*obstacles[i].coeff*p1.dist(p2)
        ans.append(total)
    return ans





def dot(p, p1, p2):
    ap=p-p1
    ab=p2-p1

    return (ap*ab)/(ab*ab)
    


def segmentProject(p, p1, p2):
    ab=p2-p1
    t=dot(p,p1,p2)


    if t<0:
        t=0
    if t>1:
        t=1

    return ab*t+p1
    
def lineProject(p, p1,p2):
    ab=p2-p1
    t=dot(p,p1,p2)

    return ab*t+p1



def distPtoSegment(p, p1,p2):
    p3=segmentProject(p,p1,p2)
    if orientation(p1.x,p1.y,p2.x,p2.y,p.x,p.y)>0:
        return abs(p3-p)
    return -abs(p3-p)



def quickhull(p1, p2, points):

    up=[]
    down=[]


    for p in points:
        val=orientation(p1.x,p1.y,p2.x,p2.y,p.x,p.y)
        if val<-1e-9:
            up.append(p)
        if val>1e-9:
            down.append(p)

    def reccusion(p1, p2, ps):
        if len(ps)==0:
            return []

        farthest=None
        maxD=0
        for p in ps:
            d=abs(p-lineProject(p,p1,p2))
            if d>maxD:
                maxD=d
                farthest=p

        

        l=[]
        r=[]
        for p in ps:
            if p is farthest: 
                continue
            if orientation(p1.x,p1.y,farthest.x,farthest.y,p.x,p.y)<-1e-9:
                l.append(p)
            if orientation(farthest.x,farthest.y,p2.x,p2.y,p.x,p.y)<-1e-9:
                r.append(p)
        return reccusion(p1,farthest,l)+[farthest]+reccusion(farthest,p2,r)

    upPath=reccusion(p1,p2,up)
    upLength=pathLength([p1]+upPath+[p2])
    downPath=reccusion(p2,p1,down)
    downPath.reverse()
    downLength=pathLength([p1]+downPath+[p2])

    return upPath if upLength<downLength else downPath
    
    

def pathLength(points):
    d=0
    for i in range(1,len(points)):
        d+=points[i].dist(points[i-1])
    return d



def inBetweenGen(p1, p2, obstacles, force=0):

    if len(obstacles)==0:
        return []


    points=[p for obstacle in obstacles for p in spread(obstacle,force)]
    if len(points)<2:
        return []
    

    return quickhull(p1,p2,points)



class Triangle(Obstacle):
    def pointIntersect(self, p):
        d1=orientation(self.points[0].x,self.points[0].y,self.points[1].x,self.points[1].y,p.x,p.y)
        d2=orientation(self.points[1].x,self.points[1].y,self.points[2].x,self.points[2].y,p.x,p.y)
        d3=orientation(self.points[2].x,self.points[2].y,self.points[0].x,self.points[0].y,p.x,p.y)

        return d1<0 and d2<0 and d3<0

class Sensor:
    def __init__(self, w=None, h=None, dir=None, r=None, a=None, coeff=None, center=None):

        self.center=Point(random.uniform(0,w),random.uniform(0,h)) if center is None else center
        self.coeff=coeff if coeff is not None else random.uniform(0.5,1)
        
        self.r=r if r is not None else random.uniform(h/20,h/10)
        self.angle=a if a is not None else random.uniform(math.pi/6,math.pi*2)
        self.dir=toDir(random.uniform(-math.pi,math.pi)) if dir is None else dir

    def thetaL(self):
        return toAngle(rotate(self.dir,self.angle/2)) *180/math.pi     
    def thetaR(self):
        return toAngle(rotate(self.dir,-self.angle/2))*180/math.pi
    
    def exposure(self, dir):
        if self.dir*dir<abs(dir)*math.cos(self.angle/2):
            return 0

        return self.coeff*math.cos(angle(dir,self.dir)/2)
        
    def intersect(self, point):
        if self.center.dist(point)>self.r:
            return False

        lm=rotate(self.dir,self.angle/2)*self.r+self.center
        rm=rotate(self.dir,-self.angle/2)*self.r+self.center
        return orientation(lm.x,lm.y,self.center.x,self.center.y,point.x,point.y)<0 and orientation(rm.x,rm.y,self.center.x,self.center.y,point.x,point.y)>0



    def casting(self, obstacles, segments):
        # segments is now a list of tuples: (index, sub_index, p1, p2)
        
        theta = math.pi/18
        lines = int(self.angle/theta)
        n = len(segments)
        costs = [0.0] * n # Use float array
        
        rays = []
        if lines % 2 == 1:
            rays.append(self.dir)
        lines -= 1

        l = rotate(self.dir, -self.angle/2)
        r = rotate(self.dir, self.angle/2)
        while lines > 0:
            rays.extend([l, r])
            lines -= 2
            l = rotate(l, theta)
            r = rotate(r, -theta)
        
        cx, cy = self.center.x, self.center.y # Cache center
        
        for ray in rays:
            # Pre-calc ray endpoint and vector
            p2 = self.center + ray * self.r
            rdx = p2.x - cx
            rdy = p2.y - cy
            
            ts = []
            
            # OPTIMIZED LOOP: No object creation, pure math
            for se in segments:
                # se[2] is p1, se[3] is p2
                t = Helper.fast_ray_segment_intersect(cx, cy, rdx, rdy, se[2].x, se[2].y, se[3].x, se[3].y)
                ts.append(t)
            
            reduce = cost(self.center, p2, obstacles, ts)
            diviend = self.exposure(ray)
            
            # Apply costs
            for i in range(n):
                if ts[i] is None:
                    continue
                # Inline distance check (optional, but faster)
                dist_sq = (ray.x*ts[i])**2 + (ray.y*ts[i])**2 
                if dist_sq > 0:
                    c = max(0, (1 - reduce[i])) * diviend / math.sqrt(dist_sq)
                    costs[i] += c

        return costs

def sensorClipping(sensor, point):
    pt=point-sensor.center
    fovPush=pt/abs(pt)*sensor.r-point

    lm=rotate(sensor.dir,sensor.angle/2)*sensor.r+sensor.center
    rm=rotate(sensor.dir,-sensor.angle/2)*sensor.r+sensor.center

    lp=segmentProject(point,sensor.center,lm)-point
    rp=segmentProject(point,sensor.center,rm)-point

    

    if abs(fovPush)<abs(lp) and abs(fovPush)<abs(rp):
        return fovPush
    if abs(lp)<abs(fovPush) and abs(lp)<abs(rp):
        return lp
    return rp

        
class Rectangle:
    def __init__(self, points):
        self.x1=1e9
        self.y1=1e9
        self.x2=-1e9
        self.y2=-1e9
        for p in points:
            self.x1=min(self.x1,p.x)
            self.y1=min(self.y1,p.y)
            self.x2=max(self.x2,p.x)
            self.y2=max(self.y2,p.y)
    
    def __repr__(self):
        return f"{self.x1} {self.y1}, {self.x2} {self.y2}"

    def intersect(self, other):
        if isinstance(other,Rectangle):
            return (self.x1<other.x2 and self.x2>other.x1 and self.y1<other.y2 and self.y2>other.y1)
        return other.x>self.x1 and other.y>self.y1 and other.x<self.x2 and other.y<self.y2
    
    def lineIntersect(self, p1, p2):
        t0=0
        t1=1
        dx=p2.x-p1.x
        dy=p2.y-p1.y

        def clip(p, q):
            nonlocal t0,t1

            if p==0:
                return q>0

            t=q/p
            if p>0:
                #leaving
                if t<t0:
                    return False
                if t<t1:
                    t1=t
            else:
                #entering
                if t>t1:
                    return False
                if t>t0:
                    t0=t
            return True
        return clip(-dx,p1.x-self.x1) and clip(dx,self.x2-p1.x) and clip(-dy,p1.y-self.y1) and clip(dy,self.y2-p1.y)



def SAT(ob1, ob2):
    axes=[]
    for i in range(len(ob1.points)):
        vec=ob1.points[(i+1)%len(ob1.points)]-ob1.points[i]
        vec=Point(vec.y,-vec.x)
        vec/=abs(vec)
        axes.append(vec)
    for i in range(len(ob2.points)):
        vec=ob2.points[(i+1)%len(ob2.points)]-ob2.points[i]
        vec=Point(vec.y,-vec.x)
        vec/=abs(vec)
        axes.append(vec)
    
    for axis in axes:
        mi1=1e9
        ma1=-1e9
        mi2=1e9
        ma2=-1e9
        for p in ob1.points:
            t=dot(p,Point(0,0),axis)
            mi1=min(t,mi1)
            ma1=max(t,ma1)
        
        for p in ob2.points:
            t=dot(p,Point(0,0),axis)
            mi2=min(t,mi2)
            ma2=max(t,ma2)
        
        if mi1>ma2 or ma1<mi2:
            return False
    return True

def triesIntersect(tries1,tries2):
    for tri1 in tries1:
        for tri2 in tries2:
            if SAT(tri1,tri2):
                return True
    return False


def pointObstalcesCollision(p, obstacles):
    #obstacles is list of tuple (obstacle, rec, tries)
    for o in obstacles:
        if o[1].intersect(p):
            if o[0].pointIntersect(p):
                return True

    return False