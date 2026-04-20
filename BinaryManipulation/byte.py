import math


class Byte:
    def __init__(self,data):
        if len(data)>8:
            data=data[-8:]

        self.bits=[0]*(8-len(data))+[int(c) if isinstance(c,str) else c for c in data]
    def __getitem__(self, index):
        return self.bits[index]
    def __setitem__(self, index, value):
        self.bits[index]=value

    def __and__(self,other):
        return Byte([a&b for a,b in zip(self.bits,other.bits)])

    def __or__(self,other):
        return Byte([a|b for a,b in zip(self.bits,other.bits)])
    def __xor__(self,other):
        return Byte([a^b for a,b in zip(self.bits,other.bits)])
    def __invert__(self):
        return Byte([(a+1)%2 for a in self.bits])
    def __lshift__(self,n):
        return Byte(self.bits[8-n:]+[0]*n)
    def __rshift__(self,n):
        return Byte([0]*n+self.bits[:8-n])
    def __repr__(self):
        ans=''
        for byte in self.bits:
            ans+=str(byte)
        return ans
    def __iter__(self):
        return iter(self.bits)

    def copy(self):
        return Byte([b for b in self.bits])

    def __len__(self):
        return 8


    def to_dec(self):
        return sum([b*(1<<i) for i,b in enumerate(reversed(self.bits))])

    def to_ascii(self):
        return chr(self.to_dec())
