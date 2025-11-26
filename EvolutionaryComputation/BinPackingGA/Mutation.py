import random
def mutation(gens, percent, gensSize):
    target=0
    newGens=[]
    while len(gens)+len(newGens)<gensSize:
        gen=gens[target].copy()
        rate=random.random()
        if rate<percent:
            i=random.randint(0,len(gens[target])-1)
            j=random.randint(0,len(gens[target])-1)
            gen[i], gen[j] = gen[j], gen[i]
        
        newGens.append(gen)
        target=(target+1)%len(gens)
    return newGens+gens