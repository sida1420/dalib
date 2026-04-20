
from point import Point

class KDTree:
    def __init__(self, split_x, points_sc, points_so):
        mid=int(len(points_sc)/2)
        self.low=None
        self.high=None
        self.split_x=split_x
        self.point=points_sc[mid]
        low=points_sc[:mid]
        high=points_sc[mid+1:]
        low_ha=set(low)
        print(low, self.point, high)

        if len(low)>0:
            self.low=KDTree(not split_x, [p for p in points_so if p in low_ha], low)
        if len(high)>0:
            self.high=KDTree(not split_x, [p for p in points_so if p not in low_ha and p!=self.point], high)

    def nearest(self, nodes, target, best, best_dist, farthest_prune=None):
        print("-----")

        dist=abs(nodes[target]-nodes[self.point])
        
        if dist<best_dist:
            best=self.point
            best_dist=dist

        if self.split_x:
            if nodes[target].x<=nodes[self.point].x:
                near=self.low
                far=self.high
            else:
                near=self.high
                far=self.low
        else:
            if nodes[target].y<=nodes[self.point].y:
                near=self.low
                far=self.high
            else:
                near=self.high
                far=self.low
        if near is not None:
            best, best_dist=near.nearest(nodes,target,best,best_dist, farthest_prune)

        if far is not None:
            if farthest_prune is None:
                farthest_prune=nodes[target]


            if self.split_x:
                farthest_prune=Point(nodes[self.point].x,farthest_prune.y)
            else:
                farthest_prune=Point(farthest_prune.x,nodes[self.point].y)

            if best_dist>=abs(farthest_prune-nodes[target]):
                best, best_dist=far.nearest(nodes,target,best,best_dist, farthest_prune)

        return best, best_dist