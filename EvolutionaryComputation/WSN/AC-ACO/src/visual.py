from matplotlib import pyplot as plt
import random
class MapView:
    def __init__(self, width, height, base_pos, nodes):
        global fig, ax

        self.base_pos=base_pos
        fig,ax = plt.subplots(figsize=(width/100, height/100))
        ax.set_box_aspect(1)
        xs=[n.x for n in nodes]
        ys=[n.y for n in nodes]

        self.nodes=ax.scatter(xs,ys,color=(0,0,1),s=2)
        ax.scatter(base_pos.x,base_pos.y,color='red',s=10)
        ax.set_xticks([])
        ax.set_yticks([])

        self.min_e=1e9
        self.max_e=0


        self.network=[]
        self.save()
    def save(self, name="map.svg"):

        plt.savefig(name)



    def clear_network(self):
        for line_list in self.network:
            for line in line_list:
                line.remove()
        self.nodes.remove()
        self.network = []

    def draw_nodes(self,nodes, residual_e, init_energy):
        colors=[(1-(e/init_energy),1-(e/init_energy),1) if e>0 else (1,0,0) for e in residual_e]
        
        xs=[n.x for n in nodes]
        ys=[n.y for n in nodes]

        self.nodes=ax.scatter(xs,ys,color=colors,s=2)

        self.save()

    def draw_network(self, nodes, e_m_list, node, thickness):
        self.min_e = min(self.min_e, min(e_m_list))
        self.max_e = max(self.max_e, max(e_m_list))

        for branch in node.branches:
            energy = e_m_list[branch.idx]

            ratio = (energy - self.min_e) / (self.max_e - self.min_e) if self.max_e > self.min_e else 0.5
            color = plt.cm.RdYlGn_r(ratio)
            sub_thickness = thickness * (0.5 + 1 * ratio)  # Thickness between 50% and 100% of the original

            if node.idx == -1:
                self.network.append(ax.plot(
                    (self.base_pos.x, nodes[branch.idx].x),
                    (self.base_pos.y, nodes[branch.idx].y),
                    color=color, linewidth=sub_thickness, zorder=-1))
            else:
                self.network.append(ax.plot(
                    (nodes[node.idx].x, nodes[branch.idx].x),
                    (nodes[node.idx].y, nodes[branch.idx].y),
                    color=color, linewidth=sub_thickness, zorder=-1))
            self.draw_network(nodes, e_m_list, branch, thickness / 2)

    