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
    M=10       # Number of ants/candidate solutions 50
    p_max=0.9  # Upper bound of the pheromone evaporation coefficient
    p_min=0.1  # Lower bound of the pheromone evaporation coefficient
    b_max=5    # Maximum heuristic information factor
    b_min=1    # Minimum heuristic information factor
    a_max=0.3  # Maximum chaotic factor
    a_min=0.05 # Minimum chaotic factor 
    t0=1       # Initial concentration of the pheromone matrix
    r=3.61     # Control parameter for Logistic chaotic mapping 
    k=5        # Sigmoid slope parameter for the adaptive heuristic weight transition
    Q=100      # Pheromone intensity constant used to control pheromone release levels  
    L_best=1000 # Globally optimal path length
    gamma=0.1 # energy consumption exponent
    pheromone_w=1 #pheromone exponent
    ctrl_bit=100 # control bit send to check communication first
    hopping_factor=0.4
    CHs_proportion=0.1
    bit_count=2000
    E_elec=50 * 1e-9            # 50 nJ/bit
    E_agg=5 * 1e-9              # 5 nJ/bit/signal
    free_space_coeff= 10 * 1e-12 # 10 pJ/bit/m^2
    multipath_coeff=0.0013 * 1e-12 # 0.0013 pJ/bit/m^4
    distance_threshold=(free_space_coeff/multipath_coeff)**0.5 
    p= 0
    beta=3
    alpha=0

    path_making_timeout=100


    with open("map.pkl",'rb') as file:
        _map=pickle.load(file)
    width, height, base_pos, nodes, init_energy, radius=_map.values()
    N=len(nodes)
    
    num_CHs=CHs_proportion*len(nodes)


    map_view=MapView(width,height,base_pos,nodes)
    

    pheromone_matrix = np.full((N, N), t0, dtype=np.float64)
    dist_matrix = [[0.0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(i):
            dist_matrix[i][j]=dist_matrix[j][i]=abs(nodes[i]-nodes[j])

    base_dists=[abs(base_pos-node) for node in nodes]

    residual_e=[init_energy]*N

    chaos=[0]*N
    chaos[0]=random.random()
    for i in range(1,N):
        chaos[i]=chaos[i-1]*(1-chaos[i-1])*r
    
    # Precompute the static transmission energy cost between all nodes
    E_m_heuristic_matrix = [[0.0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i != j:
                # Note: receive_bit=0, relay_bit=0 because we just want the Tx cost
                Em_cost = E_m(E_elec, free_space_coeff, E_agg, multipath_coeff, bit_count, 0, 0, ctrl_bit, 0, dist_matrix[i][j], distance_threshold)
                E_m_heuristic_matrix[i][j] = (1 / Em_cost) ** gamma
    #Main loop

    INF=1e9
    E_total=0
    E_lb=0
    E_ub=init_energy*N

    global_best_path=[]
    global_best_e_consumption=[[],INF]
    global_best_net=None

    live_nodes=list(range(N))
    dead_nodes=set()


    for t in range(T_max):
        print("Iteration:", t)
        p, beta, alpha=adapt(p,beta,alpha,
            t,T_max,
            p_min,p_max,
            b_min,b_max,k,
            a_min,a_max,E_total,E_lb,E_ub)
        paths=[]
        energy_consumptions_list=[]
        networks=[]

        iter_best_path_i=0


        starts=random.sample(live_nodes,min(M,len(live_nodes)))

        # virtual_energy=residual_e.copy()


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