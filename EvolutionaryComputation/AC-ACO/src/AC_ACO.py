import pickle
import random
from adaptive import adapt
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
    hopping_factor=0.4
    CHs_proportion=0.1
    bit_count=2000
    E_elec=50
    E_agg=5
    free_space_coeff=10
    multipath_coeff=0.0013
    distance_threshold=(free_space_coeff/multipath_coeff)**0.5 
    R_max=50

    p= b= a=0

    with open("map.pkl",'rb') as file:
        _map=pickle.load(file)
    width, height, base_pos, nodes, init_energy, radius=_map.values()
    N=len(nodes)
    
    num_CHs=CHs_proportion*len(nodes)

    pheromone_matrix=[[t0 for _ in range(num_nodes)] for _ in range(num_nodes)]
    dist_matrix = [[0.0 for _ in range(num_nodes)] for _ in range(num_nodes)]
    for i in range(N):
        for j in range(i):
            dist_matrix[i][j]=dist_matrix[j][i]=abs(nodes[i]-nodes[j])

    base_dists=[abs(base_pos-node) for node in nodes]

    residual_e=[init_energy]*N

    chaos=[0]*N
    chaos[0]=random.random()
    for i in range(1,N):
        chaos[i]=chaos[i-1]*(1-chaos[i-1])*r
    
    #Main loop

    for t in range(T_max):
        adapt(p,b,a,
            t,T_max,
            p_min,p_max,
            b_min,b_max,k,
            a_min,a_max,E_total,E_lb,E_ub)

        


run()
        

        

    


