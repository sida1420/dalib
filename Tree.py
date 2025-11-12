
import heapq

def pruferSequence(tree: dict[str, list[str]]):
    if len(tree) <= 2:
        return []
    ans=[]

    queue=[]
    formatedTree={}
    for label in tree.keys():
        formatedTree[label]=len(tree[label])
        if(formatedTree[label]==1):
            heapq.heappush(queue,label)



  
    while len(ans)<len(tree)-2:
        label=heapq.heappop(queue)

        neighbor=None
        for cNode in tree[label]:
            if formatedTree[cNode]>0:
                neighbor=cNode
                break

        ans.append(neighbor)
        formatedTree[label]=0
        formatedTree[neighbor]-=1

        if(formatedTree[neighbor]==1):
            heapq.heappush(queue,neighbor)


    return ans


def KruskalMST(grap: dict[str, list[tuple[str,int]]]):

    edges=[]
    for node in grap:
        for cNode, cost in grap[node]:
            edges.append((cost,node,cNode))

    edges.sort(key= lambda cost,_,_: cost)


    tree=[]

    nodes=set()
    i=0
    while len(nodes)<len(grap):
        






