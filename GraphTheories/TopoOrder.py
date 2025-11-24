
def toTopo(grap: list[list[str, list[str], list[str]]])-> list[str]:
    """
    Make topo ordered version of grap

    Parameters
    ----------
    list of grap's nodes, each nodes if a list =[char(label),list(ind),list(oud)]
    ind and oud are lists of labels

    Returns
    -------
    list of label
    """

    topoList=[]
    if len(grap)==0:
        return []
    newGrap=[]
    count=0
    for i in range(0,len(grap)):
        label, ind, outd = grap[i]
        if len(ind)==0:
            topoList.append(grap[i][0])
            count+=1


    if count==0:
        return [False]
    removed=set(topoList)

    for i in range(0,len(grap)):
        label, ind, outd= grap[i]

        if label in removed:
            continue
        newInd=[j for j in ind if j not in removed]
        newGrap.append([label,newInd,outd])

    
    return topoList + toTopo(newGrap)

from collections import deque

def toTopo2(grap: dict[str,list[tuple[str,int]]]):
    

    ind={label: 0 for label in grap}


    for oud in grap.values():
        for label, _ in oud:
            ind[label]+=1

    queue=deque([label for label in ind if ind[label]==0])

    topoList=[]

    while queue:
        label=queue.popleft()

        topoList.append(label)

        for outLabel, _ in grap[label]:
            ind[outLabel]-=1
            if ind[outLabel]==0:
                queue.append(outLabel)


    if len(topoList)!=len(grap):
        raise ValueError("Contain circle")

    return topoList

    

    

    


