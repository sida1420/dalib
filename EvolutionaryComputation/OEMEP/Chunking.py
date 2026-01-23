import math
import Classes as cl
class Map:
    def __init__(self,w,h, chunkSize):
        self.chunks=[]
        self.w=math.ceil(w/chunkSize)
        self.h=math.ceil(h/chunkSize)
        self.chunkSize=chunkSize
        for j in range(h):
            row=[]
            for i in range(w):
                row.append(Chunk())
            self.chunks.append(row)
        self.outerChunk=Chunk()

    def insertSensor(self, center, idx):
        i,j=self.whatChunk(center)
        if i<0 or j<0 or j>=self.h or i>=self.w:
            self.outerChunk.insertSensor(idx)
        else:
            self.chunks[j][i].insertSensor(idx)

    def getSensors(self, point):
        i, j=self.whatChunk(point)
        sensors=[]
        outer=False
        for jj in range(j-1,j+2):
            if jj<0 or jj>=self.h:
                outer=True
                continue
            for ii in range(i-1,i+2):
                if ii<0 or ii>=self.w:
                    outer=True
                    continue
                sensors+=self.chunks[j][i].getSensors()
        if outer:
            sensors+=self.outerChunk.getSensors()
        return sensors

    def rayCasting(self, p1, p2):
        i,j = self.whatChunk(p1)
        ti, tj=self.whatChunk(p2)
        if i==ti and j==tj:
            return self.getSensors( p1)

        d=p2-p1

        if d.x!=0:
            yx= d.y/d.x
            rx=i*self.chunkSize if d.x<0 else (i+1)*self.chunkSize
            mx=math.sqrt((p1.x-rx)**2+(yx*(p1.x-rx))**2)
            dx=math.sqrt(self.chunkSize**2+(self.chunkSize*yx)**2)
            di=int(1 if d.x>0 else -1)
        else:
            mx=math.inf
        
        if d.y!=0:
            xy= d.x/d.y
            ry=j*self.chunkSize if d.y<0 else (j+1)*self.chunkSize
            my=math.sqrt((p1.y-ry)**2+(xy*(p1.y-ry))**2)
            dy=math.sqrt(self.chunkSize**2+(self.chunkSize*xy)**2)
            dj=int(1 if d.y>0 else -1)
        else:
            my=math.inf


        cx={}

        def expand(x,y):
            if x not in cx:
                cx[x]=[y-1,y+1]
            else:
                cx[x][0]=min(cx[x][0],y-1)
                cx[x][1]=max(cx[x][1],y+1)
        

        expand(i,j)
        expand(i-1,j)
        expand(i+1,j)

        while i!=ti or j!=tj:
            if mx==my:
                i+=di
                mx+=dx
                j+=dj
                my+=dy
            elif mx>my:
                j+=dj
                my+=dy
            else:
                i+=di
                mx+=dx
            expand(i,j)
            expand(i-1,j)
            expand(i+1,j)
        
        ans=[]
        outer=False
        for x, y in cx.items():
            if x<0 or x>=self.w:
                outer=True
                continue
            for iy in range(y[0],y[1]+1):
                if iy<0 or iy>=self.h:
                    outer=True
                    continue
                ans+=self.chunks[iy][x].getSensors()
        if outer:
            ans+=self.outerChunk.getSensors()
        return ans

    def whatChunk(self,point):
        return (int(point.x//self.chunkSize),int(point.y//self.chunkSize))


    def __repr__(self):
        ans="Inner chunks:\n"
        for j in range(self.h):
            for i in range(self.w):
                ans+=f"{self.chunks[j][i]}\t"
            ans+="\n"
        ans+=f"Outer chunk: {self.outerChunk}"
        return ans



class Chunk:
    def __init__(self):
        self.sensors=[]
    def insertSensor(self, sensorIdx):
        self.sensors.append(sensorIdx)

    def getSensors(self):
        return self.sensors
    def __repr__(self):
        return f"{len(self.sensors)}"



