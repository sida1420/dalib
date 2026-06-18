import math
def sigmoid(t, T_max, k):
    
    # Midpoint becomes 0.5
    z = k * (t - T_max / 2)
    
    # Clip z to prevent overflow just in case k is very large
    z = max(-50, min(50, z))
    
    return 1 / (1 + math.exp(-z))

def adapt(p, beta, alpha, t, T_max, p_min, p_max, b_min, b_max, k, a_min, a_max, E_total, E_lb, E_ub, hopping_factor, hopping_factor_min, hopping_factor_max):
    p=p_max-(t/T_max)*(p_max-p_min)

    beta=b_min+(b_max-b_min)*sigmoid(t,T_max,k)

    alpha=a_min+(a_max-a_min)*(E_total-E_lb)/(E_ub-E_lb)

    alpha = max(a_min, min(a_max, alpha))

    hopping_factor=hopping_factor_min+(hopping_factor_max-hopping_factor_min)*(t/T_max)
    
    return p, beta, alpha, hopping_factor
