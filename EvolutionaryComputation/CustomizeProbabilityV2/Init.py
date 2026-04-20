
import random
from Vector import Vector
from Classes import Operation
import Classes
def init(n):
    one=list(Classes.one)
    two=list(Classes.two)
    choices=len(two)+len(one)

    def recursion(k):
        if k<=1:
            if random.random()<0.5:
                return Operation(None,value=random.uniform(-1,1))
            else:
                return Operation(None)
        if k==2:
            i=random.randint(0,len(one)-1)
            operator=one[i]
            return Operation(operator,[recursion(k-1)])

        i=random.randint(0,choices-1)
        
        if i<len(one):
            operator=one[i]
            return Operation(operator,[recursion(k-1)])
        else:
            i-=len(one)
            operator=two[i]
            k-=1
            split=random.randint(1,k-1)

            return Operation(operator, [recursion(split),recursion(k-split)])
        
    return recursion(n)


def init_weight_vectors(num_divisions, num_objectives):

    ans=[]

    step=1/num_divisions
    divisions=[i*step for i in range(num_divisions+1)]

    def recursion(i, sum, task):
        if i>=num_objectives:
            if abs(sum-1)<=1e-9:
                ans.append(Vector(task.copy()))
            
            return
        for j in divisions:
            if sum+j>1+1e-9:
                return
            task.append(j)
            recursion(i+1,sum+j,task)
            task.pop()

    recursion(0,0,[])
    return ans


def who_am_i_neighbour(current,weights, num_neighbours):
    neighs=[]

    dists=[]
    for idx in range(len(weights)):
        dists.append((weights[current].dist(weights[idx]),idx))
    dists.sort()

    ans=[dists[i][1] for i in range(num_neighbours)]
    return ans


def init_reference_point(evas, objectives):
    ans=[float('inf') for obj in objectives]

    for eva in evas:
        for i in range(len(objectives)):
            ans[i]=min(ans[i],eva[objectives[i]])
    return Vector(ans)
