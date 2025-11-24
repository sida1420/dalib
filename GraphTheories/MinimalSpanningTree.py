
import DisjointSet


#untested with undirected grap
def kruskalMST(grap: dict[str, list[tuple[str,int]]]):

    edges=[]
    for node in grap:
        for cNode, cost in grap[node]:
            edges.append((cost,node,cNode))

    edges.sort(key= lambda edge: edge[0])



    tree=[]
    cost=0
    djSet=DisjointSet.DisjointSet(grap.keys())
    i=0
    while len(tree)<len(grap)-1:
        minEdge=edges[i]
        i+=1
    
        nodeA=djSet.findRepresentative(minEdge[1])
        nodeB=djSet.findRepresentative(minEdge[2])

        if nodeA!=nodeB:
            djSet.union(nodeA,nodeB)
            tree.append(minEdge)
            cost+=minEdge[0]
    
    return tree, cost

import heapq

#untested with undirected grap
def primMST(grap: dict[str,list[tuple[str,int]]]):
    
    tree=[]
    totalCost=0

    randomNode=next(iter(grap)) #first element
    vertices=set([randomNode]) 
    queue=[]
    for cNode, cost in grap[randomNode]:
        heapq.heappush(queue,(cost,randomNode,cNode))


    while len(tree)<len(grap)-1:
        
        minEdge=heapq.heappop(queue)
        if minEdge[2] in vertices:
            continue

        vertices.add(minEdge[2])
        totalCost+=minEdge[0]
        tree.append(minEdge)

        for cNode, cost in grap[minEdge[2]]:
            if cNode not in vertices:
                heapq.heappush(queue,(cost, minEdge[2],cNode))



    return tree, totalCost

