from node import Node
import kdtree
    

def network_config(nodes, CHs, R_max, d0, base_pos, hopping_factor, base_dists, dist_matrix, residual_e):
    CH_nodes=[Node(idx,isCH=True) for idx in CHs]

    tree=kdtree.KDTree(True,sorted(CHs, key=lambda idx: nodes[idx].x), sorted(CHs,key=lambda idx: nodes[idx].y))

    CHs_lookup=set(CHs)
    CHs_mapping={node.idx:i for i, node in enumerate(CH_nodes)}
    for i, node in enumerate(nodes):
        if i in CHs_lookup or residual_e[i]<=0:
            continue
        best,_=tree.nearest(nodes,i,None,1e9)
        
        idx=CHs_mapping[best]
        if dist_matrix[best][i]>R_max:
            return None
        node_node=Node(i)
        CH_nodes[idx].branches.append(node_node)
        node_node.set_parent(best)
    base=Node(-1)

    for node in CH_nodes:
        dist=base_dists[node.idx]
        if dist<R_max:
            base.branches.append(node)
            node.set_parent(base.idx)
        else:
            #list all nodes closer to base and in communication range
            candidates=[cnode for cnode in CH_nodes if cnode.idx!=node.idx and dist_matrix[node.idx][cnode.idx]<R_max and dist>base_dists[cnode.idx] and residual_e[cnode.idx]>0]

            if len(candidates)==0:
                # if dist<R_max:
                #     base.branches.append(node)
                #     node.set_parent(base.idx)

                return None
                # base.branches.append(node)
                # node.set_parent(base.idx)
                # continue

            sum_candidates_e=sum([residual_e[cnode.idx] for cnode in candidates])

            
            costs=[hopping_factor*sum_candidates_e/residual_e[cnode.idx]
                +(1-hopping_factor)*(dist_matrix[node.idx][cnode.idx]**2+base_dists[cnode.idx]**2)/base_dists[node.idx]**2 for cnode in candidates]

            optimal_node_i=0

            for i, cost in enumerate(costs):
                if costs[optimal_node_i]>cost:
                    optimal_node_i=i

            candidates[optimal_node_i].branches.append(node)
            node.set_parent(candidates[optimal_node_i].idx)

    return base


def E_data_receiving(E_elec, bit_count):
    return E_elec*bit_count


def E_data_aggregating(E_agg, bit_count):
    return 0#E_agg*bit_count

def E_transmitting(E_elec, free_space_coeff, multipath_coeff, dist, d0, bit_count):
    if dist<d0:
        return bit_count*(E_elec+free_space_coeff*dist**2)
    return bit_count*(E_elec+multipath_coeff*dist**4)

def E_m(E_elec, free_space_coeff, E_agg, multipath_coeff, single_node_bit, receive_bit, relay_bit, ctrl_bit, num_branches, dist, d0):
    E_rx=E_data_receiving(E_elec,receive_bit+relay_bit+ctrl_bit*num_branches)
    E_da=E_data_aggregating(E_agg, single_node_bit+receive_bit) if receive_bit>0 else 0
    E_tx=E_transmitting(E_elec,free_space_coeff,multipath_coeff, dist, d0, relay_bit+single_node_bit+ctrl_bit)


    return E_rx+E_da+E_tx

from collections import deque

def energy_consumption(nodes, net, d0, bit_count, ctrl_bit, base_dists, dist_matrix, E_elec, E_agg, free_space_coeff, multipath_coeff):


    qu=deque([net])

    topology_net=[]

    while qu:
        top=qu.popleft()

        topology_net.append(top)
        for branch in top.branches:
            qu.append(branch)
    
    E_sum=0

    E_m_list=[0]*len(nodes)

    for i in range(len(topology_net)-1,0,-1):
        receive_bit=0
        relay_bit=0

        for branch in topology_net[i].branches:
            if branch.isCH:
                relay_bit+=branch.relay_data
            else:
                receive_bit+=bit_count

        topology_net[i].relay_data=bit_count+relay_bit if topology_net[i].isCH else 0
        dist=dist_matrix[topology_net[i].idx][topology_net[i].p_idx] if topology_net[i].p_idx!=-1 else base_dists[topology_net[i].idx]

        E=E_m(E_elec,free_space_coeff,E_agg,multipath_coeff,bit_count, receive_bit, relay_bit, ctrl_bit, len(topology_net[i].branches), dist,d0)
        E_sum+=E
        E_m_list[topology_net[i].idx]=E
    
    return E_m_list, E_sum
    

