from evaluate import E_m
import random
def make_path(num_CHs, start_node_idx, live_nodes, pheromone_matrix, dist_matrix, E_m_heuristic_matrix, residual_e, pheromone_w, beta, alpha, chaos):
    

    curr_node_idx=start_node_idx
    CH_list=[curr_node_idx]
    CH_set=set([curr_node_idx,])

    while len(CH_list)<num_CHs:
        allowed_nodes=[node_i for node_i in live_nodes if node_i not in CH_set]

        if len(allowed_nodes)==0:
            return None

        vals=[(
            pheromone_matrix[curr_node_idx][node_i]**pheromone_w
            *(residual_e[node_i]/dist_matrix[curr_node_idx][node_i])**beta
            * E_m_heuristic_matrix[curr_node_idx][node_i]
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






        