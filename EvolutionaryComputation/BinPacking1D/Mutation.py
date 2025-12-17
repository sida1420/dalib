import random


def inversion(gen):
    i=random.randint(0,len(gen)-1)
    j=i+min(random.randint(0,len(gen)-1-i),random.randint(0,len(gen)-1-i))

    while i<j:
        gen[i], gen[j]=gen[j],gen[i]
        i+=1
        j-=1

    return gen



def permutation(gen):
    i=random.randint(0,len(gen)-1)
    j=random.randint(0,len(gen)-1)
    gen[i], gen[j] = gen[j], gen[i]

    return gen



def mutation(gens, percent, gensSize):
    target=0
    newGens=[]
    while len(gens)+len(newGens)<gensSize:
        gen=gens[target].copy()
        rate=random.random()
        if rate<percent:
            gen=permutation(gen)
            # gen=inversion(gen)
        newGens.append(gen)
        target=(target+1)%len(gens)
    return newGens+gens