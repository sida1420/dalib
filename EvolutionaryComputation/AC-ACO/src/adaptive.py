
def sigmoid(t,T_max, k):
    return 1/(1+e**(-k*(t-T_max*0.5)))

def adapt(p, b, a, t, T_max, p_min, p_max, b_min, b_max, k, a_min, a_max, E_total, E_lb, E_ub):
    p=p_max-(t/T_max)*(p_max-p_min)

    b=b_min+(b_max-b_min)*sigmoid(t,T_max,k)

    a=a_min+(a_max-a_min)*(E_total-E_lb)/(E_ub-E_lb)

    

