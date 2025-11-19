
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
