

def findEulerianCycle(grap: dict[str,list[str]]):

    visitedEdges=set()


    current=next(iter(grap))
    edgesLeft=len(grap[current])
    for node in grap.keys():
        if len(grap[node])==0 or len(grap[node])%2==1:
            return set()

    
    
    def DFS(node, start):
        path=[]
        for v in grap[node]:
            if (node, v) in visitedEdges or (v,node) in visitedEdges:
                continue

            
            visitedEdges.add((node,v))
            temp=[]
            temp.append((node,v))
            if v!=start:
                temp+=DFS(v,start)
            
            start=node
            path=temp+path
        return path


    ans=DFS(current,current)
    return ans

                






grap = {
    'S': ['B', 'A'],
    'B': ['S', 'A'],
    'A': ['S', 'B', 'C', 'D'], # Order is crucial
    'C': ['A', 'D'],
    'D': ['A', 'C']
}

a=findEulerianCycle(grap)
for u,v in a:
    print(u,v)