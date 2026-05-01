from evaluate import E_m
import random
def make_path(num_CHs, start_node_idx, neighbor_nodes, pheromone_matrix, dist_matrix, residual_e, pheromone_w, beta, gamma, alpha, E_elec, free_space_coeff,E_agg,multipath_coeff, bit_count, distance_threshold, chaos):
    

    curr_node_idx=start_node_idx
    CH_list=[curr_node_idx]
    CH_set=set([curr_node_idx,])

    while len(CH_list)<num_CHs:
        allowed_nodes=[node_i for node_i in neighbor_nodes[curr_node_idx] if node_i not in CH_set and residual_e[node_i]>0]

        if len(allowed_nodes)==0:
            return None

        vals=[(
            pheromone_matrix[curr_node_idx][node_i]**pheromone_w
            *(residual_e[node_i]/dist_matrix[curr_node_idx][node_i])**beta
            *(1/E_m(E_elec,free_space_coeff,E_agg,multipath_coeff,bit_count,0,0,dist_matrix[curr_node_idx][node_i],distance_threshold))**gamma
        ) for node_i in allowed_nodes]

        sum_vals=sum(vals)

        probs=[val/sum_vals for val in vals]
        probs=[prob+chaos[curr_node_idx]*alpha for prob in probs]
        # sum_probs=sum(probs)
        # probs=[prob/sum_probs for prob in probs]

        curr_node_idx=random.choices(allowed_nodes, weights=probs,k=1)[0]
        CH_list.append(curr_node_idx)
        CH_set.add(curr_node_idx)

    return CH_list






        