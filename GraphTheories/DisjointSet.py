class DisjointSet:
    def __init__(self, nodes: list[str]):
        self.database={node:node for node in nodes}

    def findRepresentative(self,node:str):
        if(self.database[node]==node):
            return node
        else:
            return self.findRepresentative(self.database[node])

    #A and B already reps
    def unionReps(self,nodeA, nodeB):
        self.database[nodeA]=nodeB

    #A and B aren't reps
    def union(self,nodeA,nodeB):
        nodeA=self.findRepresentative(nodeA)
        nodeB=self.findRepresentative(nodeB)

        if(nodeA!=nodeB):
            self.unionReps(nodeA,nodeB)

