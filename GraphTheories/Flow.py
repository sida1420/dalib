'''
//not effecient, sub-optimal answer
def maxFlowFF(network: dict[str, list[tuple[str, int] ]], source: str, terminal: str):
    residualGraph={}

    lookupTable={}


    for s, out in network.items():
        if s not in lookupTable:
            lookupTable[s]={}
        if s not in residualGraph:
            residualGraph[s]=[]

        for v, capacity in out:
            if v not in residualGraph:
                residualGraph[v]=[]
            residualGraph[v].append(s)

            lookupTable[s][v]=capacity

            if v not in lookupTable:
                lookupTable[v]={}
            lookupTable[v][s]=0

    
    def DFS(pre, current, flow):
        if current == terminal:
            return flow
        ans=0
        pre.add(current)
        for fNode,_ in network[current]:
            if ans>=flow:
                break
            if fNode in pre or lookupTable[current][fNode]==0:
                continue
            temp=DFS(pre,fNode,min(flow-ans,lookupTable[current][fNode]))
            lookupTable[current][fNode]+=temp
            lookupTable[fNode][current]-=temp
            ans+=temp

        
        for bNode in residualGraph[current]:
            if ans>=flow:
                break
            if bNode in pre or lookupTable[current][bNode]==0:
                continue
            temp=DFS(pre,bNode,min(flow-ans,-lookupTable[current][bNode]))
            lookupTable[current][bNode]-=temp
            lookupTable[bNode][current]+=temp
            ans+=temp
        
        pre.remove(current)
        return ans
    
    return DFS(set(),source,100000)
''' 


from collections import defaultdict
def maxFlowFF(network: dict[str, list[tuple[str, int] ]], source: str, terminal: str):
    residualGraph=defaultdict(dict)
    adj=defaultdict(set)

    nodes=set(network.keys())
    for u in network.keys():
        nodes.add(u)
        for v,capacity in network[u]:
            nodes.add(v)

            residualGraph[u][v]=residualGraph[u].get(v,0) +capacity
            residualGraph[v][u]=0

            adj[u].add(v)
            adj[v].add(u)


    maxFlow=0

    for node in nodes:
        residualGraph.setdefault(node,{})
        adj.setdefault(node,set())

    INF=10**12

    def DFS(current: str, visited: set, flow: int):
        
        if current==terminal:
            return flow
        visited.add(current)
        for target in adj[current]:
            if target in visited or residualGraph[current][target]<=0:
                continue

            pushed=DFS(target,visited,min(flow,residualGraph[current][target]))

            if pushed:
                residualGraph[current][target]-=pushed
                residualGraph[target][current]+=pushed

                return pushed
        return 0


    while True:
        
        pushed=DFS(source,set(),INF)

        if pushed:
            maxFlow+=pushed
        else:
            break

    return maxFlow


#TO DO: BFS type

network = {
    "s": [("a", 16), ("c", 13)],
    "a": [("b", 12)],
    "b": [("c", 9), ("t", 20)],
    "c": [("a", 4), ("d", 14)],
    "d": [("b", 7), ("t", 4)],
    "t": []
}


print(maxFlowFF(network,"s","t"))