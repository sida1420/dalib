import pickle
from updater import *
import random
from adaptive import adapt
from path_constructing import make_path
from evaluate import *
import numpy as np
from visual import MapView
import pandas as pd
def run():


    #Initilize

    #hyterparameters
    T_max=3500 # The maximum number of iterations used in simulation
    M=50       # Number of candidate solutions
    ctrl_bit=100 # control bit send to check communication first
    hopping_factor=0.4
    CHs_proportion=0.1
    bit_count=2000
    E_elec=50 * 1e-9            # 50 nJ/bit
    E_agg=5 * 1e-9              # 5 nJ/bit/signal
    free_space_coeff= 10 * 1e-12 # 10 pJ/bit/m^2
    multipath_coeff=0.0013 * 1e-12 # 0.0013 pJ/bit/m^4
    distance_threshold=(free_space_coeff/multipath_coeff)**0.5 
    c1=0.8 #cognitive component
    c2=0.9 #sosical component
    w=0.5 #initial weight (current velocity decay)



    path_making_timeout=100


    with open("map.pkl",'rb') as file:
        _map=pickle.load(file)
    width, height, base_pos, nodes, init_energy, radius=_map.values()
    N=len(nodes)
    
    num_CHs=int(CHs_proportion*len(nodes))


    map_view=MapView(width,height,base_pos,nodes)
    

    dist_matrix = [[0.0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(i):
            dist_matrix[i][j]=dist_matrix[j][i]=abs(nodes[i]-nodes[j])

    base_dists=[abs(base_pos-node) for node in nodes]

    residual_e=[init_energy]*N
    
    # # Precompute the static transmission energy cost between all nodes
    # E_m_heuristic_matrix = [[0.0 for _ in range(N)] for _ in range(N)]
    # for i in range(N):
    #     for j in range(N):
    #         if i != j:
    #             # Note: receive_bit=0, relay_bit=0 because we just want the Tx cost
    #             Em_cost = E_m(E_elec, free_space_coeff, E_agg, multipath_coeff, bit_count, 0, 0, ctrl_bit, 0, dist_matrix[i][j], distance_threshold)
    #             E_m_heuristic_matrix[i][j] = (1 / Em_cost) ** gamma

    #Initilize swam
    particles=[]
    best_nets=[]
    
    for i in range(M):
        dummy_particle=np.random.randint(0,N,size=(num_CHs))
        while True:
            dummy_net=network_config(nodes,dummy_particle.copy(),radius,distance_threshold,base_pos,hopping_factor,base_dists,dist_matrix,residual_e)
            if dummy_net is not Node:
                break
        particles.append(dummy_particle)
        best_nets.append(dummy_net)

        
    
    best_Es=[energy_consumption(particles[i].tolist(),best_nets[i],distance_threshold,bit_count,ctrl_bit,base_dists,dist_matrix,E_elec,E_agg,free_space_coeff,multipath_coeff) for i in range(M)]
    best_positions=[particles[i].copy() for i in range(M)]
    velocities=[np.zeros((num_CHs)) for i in range(M)]
    
    INF=1e9
    E_total=0
    E_lb=0
    E_ub=init_energy*N

    global_best_path=[]
    global_best_e_consumption=[[],INF]
    global_best_net=None

    with np.argmin(best_Es) as min_i:
        global_best_path=particles[min_i].copy()
        global_best_e_consumption=best_Es[min_i][1]
        global_best_net=best_nets[min_i]


    live_nodes=list(range(N))
    dead_nodes=set()

    #Main loop
    

    for t in range(T_max):
        print("Iteration:", t)

        paths=[]
        energy_consumptions_list=[]
        networks=[]

        iter_best_path_i=0

        #MOVE PARTICLES        
        for i in range(M):
            r1=np.random.rand(num_CHs)
            r2=np.random.rand(num_CHs)

            cognitive_component=r1*c1*(best_positions[i]-particles[i])
            social_component=r2*c2*(global_best_path-particles[i])
            particles[i]=w*particles[i] + cognitive_component+social_component

            




        for ant in range(len(starts)):
            timeout=0
            while timeout<path_making_timeout:
                path=make_path(
                    num_CHs,starts[ant],live_nodes,pheromone_matrix,dist_matrix,E_m_heuristic_matrix,residual_e,pheromone_w,beta,alpha,chaos
                )
                if path is None: 
                    timeout+=1
                    continue
                
                net=network_config(
                    nodes,path, radius, distance_threshold, base_pos, hopping_factor, base_dists, dist_matrix, residual_e
                )
                if net is None:
                    timeout+=1
                    continue
                

                e_m_list, e_sum=energy_consumption(nodes, net, distance_threshold, bit_count, ctrl_bit, base_dists, dist_matrix, E_elec, E_agg, free_space_coeff, multipath_coeff)
                break
            if timeout>=path_making_timeout: #can't find a path for this ant
                continue
            # virtual_energy=[energy-consume_amount for energy, consume_amount in zip(virtual_energy, e_m_list)]
            pheromone_update(pheromone_matrix,Q,path, chaos, alpha, dist_matrix)            
            
            paths.append(path)
            energy_consumptions_list.append((e_m_list, e_sum))
            networks.append(net)

            if energy_consumptions_list[iter_best_path_i][1] > e_sum:
                iter_best_path_i=len(paths)-1

        if len(paths)==0: # can't find a path within iteration
            print("No path found in iteration ", t)
            break

        if energy_consumptions_list[iter_best_path_i][1]<global_best_e_consumption[1]:
            global_best_e_consumption=energy_consumptions_list[iter_best_path_i]
            global_best_path=paths[iter_best_path_i]
            global_best_net=networks[iter_best_path_i]
        
        E_total+=energy_consumptions_list[iter_best_path_i][1]
        residual_e=[max(0, energy-consume_acount) for energy, consume_acount in zip(residual_e,energy_consumptions_list[iter_best_path_i][0])]

        new_dead=0
        i=0
        while i < len(live_nodes):
            j = live_nodes[i]
            e = residual_e[j]
            if e <= 0:
                dead_nodes.add(j)
                live_nodes[i], live_nodes[-1]=live_nodes[-1], live_nodes[i]
                live_nodes.pop()
                new_dead+=1
                i-=1
            i+=1




        L_best=min(L_best, path_length(paths[iter_best_path_i], dist_matrix))
        pheromone_update(pheromone_matrix,Q,paths[iter_best_path_i], chaos, alpha, dist_matrix) #Eq 12

        for i in range(0,N):
            chaos[i]=chaos[i]*(1-chaos[i])*r


        pheromone_matrix = np.clip(pheromone_matrix * (1 - p), 0.1, 10.0)

        print([round(e,2) for e in residual_e])
        print("Best path: ", paths[iter_best_path_i])
        print("Minimum energy comsumption: ",round(energy_consumptions_list[iter_best_path_i][1],3))
        print(f"Adaptive: p={round(p,2)}, beta={round(beta,2)}, alpha= {round(alpha,2)}")
        print("New dead node: ", new_dead,"/",len(dead_nodes))

        new_entry=pd.DataFrame([{
            "round": t+1,
            "alive_nodes": len(live_nodes),
            "round_energy": energy_consumptions_list[iter_best_path_i][1]}])

        new_entry.to_csv("src/round_state.csv", mode='a', header=False, index=False)

        if t%100==0:
            map_view.clear_network()
            map_view.draw_nodes(nodes,residual_e, init_energy)
            map_view.draw_network(nodes, energy_consumptions_list[iter_best_path_i][0], networks[iter_best_path_i], 1)
            map_view.save(f"configurations/config{t}.svg")




run()