
import random
from Vector import Vector
def init(n):
    return [random.uniform(-2,2) for i in range(n)]



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
