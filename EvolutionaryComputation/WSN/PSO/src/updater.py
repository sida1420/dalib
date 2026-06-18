

def pheromone_update(pheromone_matrix, Q, path, chaos, alpha, dist_matrix):
    L = path_length(path, dist_matrix)
    for i in range(len(path)-1):
        pheromone_matrix[path[i]][path[i+1]] += Q / L + alpha * chaos[path[i]]

def path_length(path, dist_matrix):
    return sum([dist_matrix[path[i]][path[i+1]] for i in range(len(path)-1)])