

def pheromone_update(pheromone_matrix, Q ,L_best, path, chaos, alpha):
    
    for i in range(len(path)-1):
        pheromone_matrix[path[i]][path[i+1]]+=Q/L_best+alpha*chaos[path[i]]