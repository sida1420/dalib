
from collections import deque

def shortestPathBFS(grap: dict[str,list[str]],startNode: str):
    
    ans = {label: float('inf') for label in grap}
    ans[startNode] = 0

    queue=deque(startNode)

    marked=set(startNode)

    while queue:
        label=queue.popleft()

        for outLabel in grap[label]:
            if outLabel not in marked:
                queue.append(outLabel)
                ans[outLabel]=ans[label]+1
                marked.add(outLabel)

    return ans





import heapq

def dijkstra(grap: dict[str, list[tuple[str,int]]], startNode: str):
    ans = {label: float('inf') for label in grap}
    ans[startNode] = 0

    heap = [(0, startNode)]

    while heap:
        dist, label = heapq.heappop(heap)

        if dist>ans[label]:
            continue

        for outLabel, cost in grap[label]:
            if ans[label] + cost < ans[outLabel]:
                ans[outLabel] = ans[label] + cost
                heapq.heappush(heap, (ans[outLabel], outLabel))
    
    return ans

import TopoOrder as to

def shortestPathAcyclic(grap: dict[str,list[tuple[str,int]]],startNode: str):
    topoList=to.toTopo2(grap)

    ans = {label: float('inf') for label in topoList}

    ans[topoList[0]]=0

    for i in range(0,len(topoList)):
        label=topoList[i]

        for outLabel,cost in grap[label]:
            ans[outLabel]=min(ans[outLabel],ans[label]+cost)

    return ans


