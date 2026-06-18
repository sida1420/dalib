class Point:
    def __init__(self,x, y):
        self.x=x
        self.y=y
    def __sub__(self, other):
        return Point(self.x-other.x,self.y-other.y)
    def __add__(self, other):
        return Point(self.x+other.x,self.y+other.y)
    def __abs__(self):
        return (self.x**2+self.y**2)**0.5
    def __truediv__(self, num):
        return Point(self.x/num,self.y/num)
    def __mul__(self, num):
        return Point(self.x*num,self.y*num)
    
    def __repr__(self):
        return f'({round(self.x,2)}, {round(self.y,2)})'