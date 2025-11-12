def genericSearch(grap: dict[str, list[str]], startNode: str):
    queue = [startNode]
    marked=set(startNode)
    while len(queue) != 0:
        label = queue.pop()
        marked.add(label)

        for outLabel in grap[label]:
            if outLabel not in marked:
                queue.append(outLabel)


def DFS(grap: dict[str, list[str]], current: str, marked: dict[str, bool]):
    if current in marked:
        return
    marked[current]=True


    for label in grap[current]:
        DFS(grap,label,marked)

from collections import deque

def BFS(garp: dict[str, list[str]], start: str):
    queue  = deque(start)

    marked=set(start)

    while len(queue)!=0:
        label= queue.popleft()

        for outLabel in grap[label]:
            if outLabel not in marked:
                queue.append(outLabel)



grap = {"a": ["b", "c"], "b": ["a"], "c": ["a"]}

DFS(grap,"a",dict())
BFS(grap,"a")
GenericSearch(grap,"a")