import math
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
        i=int(center.x//self.chunkSize)
        j=int(center.y//self.chunkSize)
        if i<0 or j<0 or j>=self.h or i>=self.w:
            self.outerChunk.insertSensor(idx)
        else:
            self.chunks[j][i].insertSensor(idx)

    def getSensors(self, point):
        i=int(point.x//self.chunkSize)
        j=int(point.y//self.chunkSize)
        sensors=[]
        for jj in range(j-1,j+2):
            for ii in range(i-1,i+2):
                if ii<0 or jj<0 or jj>=self.h or ii>=self.w:
                    sensors+=self.outerChunk.getSensors()
                else:
                    sensors+=self.chunks[j][i].getSensors()
        return sensors

    def __repr__(self):
        ans=""
        for j in range(self.h):
            for i in range(self.w):
                ans+=f"{self.chunks[j][i]}\t"
            ans+="\n"
        ans+=f"{self.outerChunk}"
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