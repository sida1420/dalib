from matplotlib import pyplot as plt
import random
class MapView:
    def __init__(self, width, height, base_pos, nodes):
        global fig, ax
        fig,ax = plt.subplots(figsize=(width/100, height/100))
        ax.set_box_aspect(1)
        ax.scatter(base_pos.x,base_pos.y,color='red',s=10)
        xs=[n.x for n in nodes]
        ys=[n.y for n in nodes]

        ax.scatter(xs,ys,color='blue',s=2)
        ax.set_xticks([])
        ax.set_yticks([])

        self.network=[]
        self.save()
    def save(self):

        plt.savefig("map.svg")

    def clear_network(self):
        self.network=[]

    def draw_network(self,nodes, node, thickness):
        color=(random.random(),random.random(),random.random())
        for branch in node.branches:
            self.network.append(ax.plot((nodes[node.idx].x,nodes[branch.idx].x),(nodes[node.idx].y,nodes[branch.idx].y),color=color,linewidth=thickness,zorder=-1))
            self.draw_network(nodes, branch, thickness/2)

    