import math
from byte import Byte
class Sbyte:
    def __init__(self,data):
        self.bytes=data


    def __getitem__(self,index):
        return self.bytes[index]

    def get_atomic(self, index):
        return self.bytes[index//8][index%8]
    def set_atomic(self, index,value):
        self.bytes[index//8][index%8]=value

    def __setitem__(self,index, value):
        self.bytes[index]=value

    def append(self, byte):
        self.bytes.append(byte)

    def __len__(self):
        return len(self.bytes)

    def atomic_len(self):
        return len(self)*8

    def __xor__(self,other):
        if len(self)<len(other):
            return other^self
        offset=len(self)-len(other)
        res=Sbyte([self.bytes[i].copy() for i in range(offset)])
        leftover=Sbyte(self.bytes[offset:])
        


        return res+Sbyte([x^y for x,y in zip(leftover,other)])

    def __add__(self, other):
        return Sbyte(self.bytes+other.bytes)

    def __iter__(self):
        return iter(self.bytes)


    def __repr__(self):
        ans=''
        for byte in self.bytes:
            ans+=str(byte)+' '
        return ans[:-1]


def dec_to_bytes(dec):
    if dec==0:
        m=0
    else:
        m=int(math.log2(dec))


    s=''
    p=1<<m
    while p>0:
        if dec>=p:
            s+='1'
            dec-=p
        else:
            s+='0'
        p>>=1

    full=(m+1)>>3
    leftover=(m+1)-(full<<3)
    return Sbyte(([Byte('0'*(8-leftover)+s[:leftover]),] if leftover!=0 else [])+[Byte([s[leftover+(i<<3)+j] for j in range(8)]) for i in range(full)])

def hex_to_dec(hex):
    dec=0
    for i, h in enumerate(hex):
        if '0'<=h<='9':
            d=int(h)
        elif 'a'<=h<='z':
            d=ord(h)-87
        else:
            d=ord(h)-55
        dec<<=4
        dec+=d
    return dec


def hex_to_bytes(hex):
    ans=None
    i=0
    if len(hex)%2!=0:
        hex='0'+hex
    while i<len(hex):
        if ans is None:
            ans=dec_to_bytes(hex_to_dec(hex[i:i+2]))
        else:
            ans+=dec_to_bytes(hex_to_dec(hex[i:i+2]))
        i+=2
    return ans

def ascii_to_bytes(ascii):
    ans=None
    i=0
    while i<len(ascii):
        if ans is None:
            ans=dec_to_bytes(ord(ascii[i]))
        else:
            ans+=dec_to_bytes(ord(ascii[i]))
        i+=1
    return ans
            