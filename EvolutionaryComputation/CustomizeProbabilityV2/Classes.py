

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
    def copy(self):
        return Point(self.x,self.y)

def repair(ind):
    for i in range(len(ind)):
        ind[i]=max(0,min(ind[i],1))
    return ind
import math
two=set(['+','-','*','/','^','max','min'])
one=set(['neg','exp','log'])

class Operation:
    def __init__(self, operator, pre=None, value=None):
        if pre is None:
            pre=[]
        self.oper=operator
        self.set_pre(pre)
        self.is_variable=False
        if operator is None:
            if value is None:
                self.is_variable=True
            else:
                self.val=value
    def __call__(self, x):
        if self.oper is None:
            if self.is_variable:
                return x
            return self.val
        

        a1=self.pre[0](x)
        if a1 is None:
            return None
        
        if self.oper=='neg':
            return -a1
        elif self.oper=='exp':
            if a1 > 709: 
                return None 
            try:
                return math.exp(a1)
            except OverflowError:
                return None
        elif self.oper=='log':
            if a1<=0:
                return None
            return math.log(a1)

        a2=self.pre[1](x)
        if a2 is None:
            return None

        if self.oper=='+':
            try:
                return a1 + a2
            except OverflowError:
                return None
        elif self.oper=='-':
            return a1-a2
        elif self.oper=='*':
            try:
                return a1 * a2
            except OverflowError:
                return None
        elif self.oper=='/':
            if a2 == 0:
                return None
            try:
                return a1 / a2
            except OverflowError:
                return None
        elif self.oper=='^':
            if a1==0 and a2<=0:
                return None
            if a1<0 and not a2.is_integer():
                return None
            try:
                return a1 ** a2
            except OverflowError:
                return None
        elif self.oper=='min':
            return min(a1,a2)
        elif self.oper=='max':
            return max(a1,a2)


    def fancy_string(self):
        if self.oper is None:
            if self.is_variable:
                return "x"
            return f"{round(self.val,2)}"
        if self.oper in one:
            if self.oper=='neg':
                return f"-({self.pre[0].fancy_string()})"
            elif self.oper=='exp':
                return f"exp({self.pre[0].fancy_string()})"
            elif self.oper=='log':
                return f"log({self.pre[0].fancy_string()})"
        if self.oper=='+':
            return f"({self.pre[0].fancy_string()})+({self.pre[1].fancy_string()})"
        elif self.oper=='-':
            return f"({self.pre[0].fancy_string()})-({self.pre[1].fancy_string()})"
        elif self.oper=='*':
            return f"({self.pre[0].fancy_string()})*({self.pre[1].fancy_string()})"
        elif self.oper=='/':
            return f"({self.pre[0].fancy_string()})/({self.pre[1].fancy_string()})"
        elif self.oper=='^':
            return f"({self.pre[0].fancy_string()})({self.pre[1].fancy_string()})"
        elif self.oper=='max':
            return f"max({self.pre[0].fancy_string()},{self.pre[1].fancy_string()})"
        elif self.oper=='min':
            return f"min({self.pre[0].fancy_string()},{self.pre[1].fancy_string()})"
        

    def set_pre(self, pre):
        self.pre=pre
        self.pre_size=0
        for p in pre:
            self.pre_size+=p.pre_size+1


    def get_node(self, i):
        if i==0:
            return self
        i-=1
        for p in self.pre:
            if i-p.pre_size-1<0:
                return p.get_node(i)
            i-=self.pre_size+1

    def update_pre(self):
        self.pre_size=0
        for p in self.pre:
            self.pre_size+=1+p.update_pre()
        return self.pre_size

        
    def __repr__(self):
        if self.oper is None:
            if self.is_variable:
                return "x"
            return f"{round(self.val,2)}"
        if self.oper in one:
            return f"[{self.oper} {self.pre[0]}]"
        return f"[{self.oper} {self.pre[0]} {self.pre[1]}]"
        

    def copy(self):

        if self.oper is None:
            if self.is_variable:
                return Operation(None)
            return Operation(None,value=self.val)
        
        return Operation(self.oper,[p.copy() for p in self.pre])

    def get_list(self):
        ans=[self]

        for p in self.pre:
            ans+=p.get_list()

        return ans

    def parsing(string):
        

        def recursion(i):
            if string[i]=='x':
                return Operation(None), i+1
            if string[i]=='-' or string[i].isdigit():
                num=string[i]
                i+=1
                while string[i] !=']' and string[i]!=' ':
                    num+=string[i]
                    i+=1
                return Operation(None,value=float(num)), i
            open_bracket=1
            i+=1

            operator=''
            while string[i]!=' ':
                operator+=string[i]
                i+=1
            i+=1
            pre0, j=recursion(i)
            if operator in one:
                return Operation(operator,[pre0]), j+1
            else:
                pre1, k=recursion(j+1)
                return Operation(operator,[pre0,pre1]), k+1
        return recursion(0)[0]


