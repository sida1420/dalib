import random


def inversion(gen):
    n=len(gen)
    i=random.randint(0,n-2)
    j=random.randint(i+1,min(i+n//5,n-1))

    while i<j:
        gen[i], gen[j]=gen[j],gen[i]
        i+=1
        j-=1

    return gen



def permutation(gen, percent):
    
    for i in range(len(gen)):
        if random.random()>percent:
            continue
        j=random.randint(0,len(gen)-1)
        gen[i], gen[j] = gen[j], gen[i]

    return gen

def chunkMove(gen):
    n = len(gen)
    a = random.randint(0, n-2)
    b = random.randint(a+1, min(a+n//5,n-1))
    chunk = gen[a:b]

    del gen[a:b]
    pos = random.randint(0, len(gen))
    gen[pos:pos] = chunk
    return gen


def mutation(gens, percent, gensSize):
    newGens=[]

    target=len(gens)
    mutateOrder=[i for i in range(len(gens))]

    while len(newGens)<gensSize:
        
        if(target>=len(gens)):
            target=0
            random.shuffle(mutateOrder)

        gen=gens[mutateOrder[target]].copy()


        if random.random()<percent:

            rate=random.random()
            if rate<0.5:
                gen=permutation(gen,0.05)
            elif rate<0.75:
                gen=chunkMove(gen)
            else:
                gen=inversion(gen)

        newGens.append(gen)
        target+=1
    return newGens