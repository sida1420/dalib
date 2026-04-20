import random

def jitter(ind):
    options=[node for node in ind.get_list() if node.oper is None and not node.is_variable]
    if len(options)==0:
        return
    options[random.randint(0,len(options)-1)].val+=random.uniform(-1,1)

import Classes

def change_operator(ind):

    options=[node for node in ind.get_list() if node.oper is not None]
    if len(options)==0:
        return
    i=random.randint(0,len(options)-1)

    if options[i].oper in Classes.one:
        options[i].oper=random.choice(list(Classes.one))
    else:
        options[i].oper=random.choice(list(Classes.two))
        
def add_operator(ind, max_complexity):
    if ind.pre_size+1>=max_complexity:
        return

    options=[node for node in ind.get_list() if node.oper is None]

    if len(options)==0:
        return

    node=random.choice(options)

    if random.choice([1,2])==1 or ind.pre_size+2==max_complexity:
        node.oper=random.choice(list(Classes.one))
        if node.is_variable:
            node.pre=[Classes.Operation(None)]
        else:
            node.pre=[Classes.Operation(None,value=node.val)]
            node.val=None
        node.pre_size=1
    else:
        node.oper=random.choice(list(Classes.two))
        pre=Classes.Operation(None) if random.random()<0.75 else Classes.Operation(None,value=random.uniform(-1,1))
        if node.is_variable:
            node.is_variable=False
            node.pre=[Classes.Operation(None),pre]
            random.shuffle(node.pre)
        else:
            node.pre=[Classes.Operation(None,value=node.val),pre]
            random.shuffle(node.pre)
            node.val=None
        node.pre_size=2

def cut_branch(ind):

    options=[node for node in ind.get_list() if node.oper is not None]
    if len(options)==0:
        return

    node=random.choice(options)
    pre=Classes.Operation(None) if random.random()<0.5 else Classes.Operation(None,value=random.uniform(-1,1))

    if len(node.pre)==1:
        node.pre=[pre]
    else:
        node.pre[random.choice([0,1])]=pre

def change_leaf(ind):
    options=[node for node in ind.get_list() if node.oper is None]
    if len(options)==0:
        return

    node=random.choice(options)
    if node.is_variable:
        node.is_variable=False
        node.val=random.uniform(-1,1)
    else:
        node.is_variable=True
        node.val=None


def mutate(ind,max_complexity):
    
    r=random.random()

    if r<0.4:
        jitter(ind)
    elif r<0.6:
        change_operator(ind)
        ind.update_pre()
    elif r<0.65:
        add_operator(ind,max_complexity)
        ind.update_pre()
    elif r<0.7:
        cut_branch(ind)
        ind.update_pre()
    else:
        change_leaf(ind)


    return ind
