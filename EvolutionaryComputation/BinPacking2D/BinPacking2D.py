

def binPacking(boxes:list, bin: int):

    #How many in each generation
    gensSize=100
    #Loop
    end=100
    bestScore=len(boxes)
    bestGen=[]
    
    gens=[Initialize.init(boxes) for i in range(0,gensSize)]

    for i in range(0,end):
        # print("\tNEW GEN")
        score, gens=Selection.selection(gens,0.4,boxes,bin)
        if bestScore>=score:
            bestScore=score
            bestGen=gens[0]
        

        gens=Crossover.crossover(gens,1)
        gens=Mutation.mutation(gens,1, gensSize)

    print("\tBEST\n",bestScore,bestGen)