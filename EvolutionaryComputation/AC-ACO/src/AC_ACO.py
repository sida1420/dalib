import pickle
from updater import *
import random
from adaptive import adapt
from path_constructing import make_path
from evaluate import *
from visual import MapView
def run():

    #Initilize

    #hyterparameters
    T_max=2500 # The maximum number of iterations used in simulation
    M=50       # Number of ants/candidate solutions 
    p_max=0.9  # Upper bound of the pheromone evaporation coefficient
    p_min=0.1  # Lower bound of the pheromone evaporation coefficient
    b_max=3    # Maximum heuristic information factor
    b_min=1    # Minimum heuristic information factor
    a_max=1    # Maximum chaotic factor
    a_min=0.1  # Minimum chaotic factor 
    t0=1       # Initial concentration of the pheromone matrix
    r=3.58     # Control parameter for Logistic chaotic mapping 
    k=0.1      # Sigmoid slope parameter for the adaptive heuristic weight transition
    Q=1        # Pheromone intensity constant used to control pheromone release levels  
    L_best=1000 # Globally optimal path length
    gamma=0.1 # energy consumption exponent
    pheromone_w=1 #pheromone exponent
    hopping_factor=0.4
    CHs_proportion=0.1
    bit_count=2000
    E_elec=50 * 1e-9            # 50 nJ/bit
    E_agg=5 * 1e-9              # 5 nJ/bit/signal
    free_space_coeff= 10 * 1e-12 # 10 pJ/bit/m^2
    multipath_coeff=0.0013 * 1e-12 # 0.0013 pJ/bit/m^4
    distance_threshold=(free_space_coeff/multipath_coeff)**0.5 
    p= beta= alpha=0

    path_making_timeout=100


    with open("map.pkl",'rb') as file:
        _map=pickle.load(file)
    width, height, base_pos, nodes, init_energy, radius=_map.values()
    N=len(nodes)
    
    num_CHs=CHs_proportion*len(nodes)


    map_view=MapView(width,height,base_pos,nodes)
    

    pheromone_matrix=[[t0 for _ in range(N)] for _ in range(N)]
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
    
    neighbor_nodes=[[node_j for node_j in range(N) if node_i!=node_j and dist_matrix[node_i][node_j]<=radius] for node_i in range(N)]
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
        print(t)
        p, beta, alpha=adapt(p,beta,alpha,
            t,T_max,
            p_min,p_max,
            b_min,b_max,k,
            a_min,a_max,E_total,E_lb,E_ub)
        paths=[]
        energy_consumptions_list=[]

        iter_best_path=[] 
        iter_best_e_consumption=[[],INF]
        iter_best_net=None


        starts=random.sample(live_nodes,min(M,len(live_nodes)))

        # virtual_energy=residual_e.copy()
        for ant in range(M):
            timeout=0
            while timeout<path_making_timeout:
                path=make_path(
                    num_CHs,starts[ant],neighbor_nodes,pheromone_matrix,dist_matrix,residual_e,pheromone_w,beta,gamma,alpha,E_elec,free_space_coeff,E_agg,multipath_coeff,bit_count,distance_threshold,chaos
                )
                if path is None: 
                    timeout+=1
                    continue
                
                net=network_config(
                    nodes,path, radius, base_pos, hopping_factor, base_dists, dist_matrix, residual_e
                )
                if net is None:
                    timeout+=1
                    continue


                e_m_list, e_sum=energy_consumption(nodes, net, distance_threshold, bit_count, base_dists, dist_matrix, E_elec, E_agg, free_space_coeff, multipath_coeff)
                break
            if timeout>=path_making_timeout: #can't find a path for this ant
                continue
            # virtual_energy=[energy-consume_amount for energy, consume_amount in zip(virtual_energy, e_m_list)]
            pheromone_update(pheromone_matrix,Q,L_best,path, chaos, alpha)            
            
            paths.append(path)
            energy_consumptions_list.append(e_sum)

            if e_sum<iter_best_e_consumption[1]:
                iter_best_e_consumption=(e_m_list, e_sum)
                iter_best_path=path
                iter_best_net=net

        if len(paths)==0: # can't find a path within iteration
            break

        if iter_best_e_consumption[1]<global_best_e_consumption[1]:
            global_best_e_consumption=iter_best_e_consumption
            global_best_path=iter_best_path
            global_best_net=iter_best_net
        
        E_total+=iter_best_e_consumption[1]
        residual_e=[energy-consume_acount for energy, consume_acount in zip(residual_e,iter_best_e_consumption[0])]

        new_dead=0
        for i,j in enumerate(live_nodes):
            e=residual_e[j]
            if e<=0:
                dead_nodes.add(j)
                live_nodes[i], live_nodes[-1]=live_nodes[-1], live_nodes[i]
                live_nodes.pop()
                new_dead+=1

        # for node_i in global_best_path:
        #     if node_i in dead_nodes:
        #         global_best_path=[]
        #         global_best_e_consumption=[[],INF]
        #         global_best_net=None
        # if global_best_net is not None and new_dead>0:
        #     global_best_net=network_config(nodes, global_best_path, radius, base_pos, hopping_factor,base_dists, dist_matrix, residual_e)
        #     global_best_e_consumption=energy_consumption(nodes, global_best_net, distance_threshold, bit_count, base_dists, dist_matrix, E_elec, E_agg, free_space_coeff, multipath_coeff)
        


        L_best=min(L_best,iter_best_e_consumption[1])
        pheromone_update(pheromone_matrix,Q,L_best,iter_best_path, chaos, alpha) #Eq 12

        for i in range(0,N):
            chaos[i]=chaos[i]*(1-chaos[i])*r


        for i in range(N):
            for j in range(N):
                pheromone_matrix[i][j] *= (1 - p)

        print([round(e,2) for e in residual_e])
        print("Best path: ", iter_best_path)
        print("Minimum energy comsumption: ",iter_best_e_consumption[1])
        print(f"Adaptive: p={round(p,2)}, beta={round(beta,2)}, alpha= {round(alpha,2)}")
        print("New dead node: ", new_dead,"/",len(dead_nodes))

    map_view.clear_network()
    map_view.draw_network(nodes, global_best_net, 1)
    map_view.save()




run()