from point import Point
    

import random

def gen(width, height, num_nodes, energy, radius,base_pos=None, num_cluster_points=0):
    if base_pos is None:
        base_pos=Point(width/2,height/2)

    nodes=[Point(random.uniform(0,width),random.uniform(0,height)) for i in range(num_nodes)]


    cluster_points=[Point(random.uniform(0,width),random.uniform(0,height)) for i in range(num_cluster_points)]

    if num_cluster_points!=0:
        for i in range(len(nodes)):
            shortest=abs(cluster_points[0]-nodes[i])
            p=cluster_points[0]
            for point in cluster_points:
                canditate=abs(point-nodes[i])
                if canditate<shortest:
                    p=point
                    shortest=canditate
            
            nodes[i]+=(p-nodes[i])*random.uniform(0,0.5)

    

    map_={"width":width,"height":height,"base_pos":base_pos,"nodes":nodes, "init_energy": energy, "radius":radius}

    return map_

import pickle
import visual
def new_map():

    map_=gen(250,250,100,0.6,100,num_cluster_points=3)
    visual.MapView(map_["width"],map_["height"],map_["base_pos"],map_["nodes"])
    with open("map.pkl", "wb") as file:
        pickle.dump(map_,file)


new_map()
