


import Init
import Selection
import Crossover
import time
from Classes import Point
import Evaluate
import Visual
import MakeTarget
objectives=("errors","complexity","gradient_error","different")


def run():
    #hyper parameters

    num_divisions=5
    num_objectives=4
    max_complexity=30
    num_neighbours=5
    replace_limit=3
    iter_limit=10000
    
    #Initilize
    target=MakeTarget.connect(MakeTarget.clamp([Point(0,0.5),Point(1280,0.95),Point(2048,1),Point(2816,0.95),Point(4096,0.5),Point(5888,0.1),Point(9000,0.01)]))

    weights=Init.init_weight_vectors(num_divisions,num_objectives)

    popu_size=len(weights)
    population=[Init.init(15) for i in range(popu_size)]
    neighbours=[Init.who_am_i_neighbour(i,weights,num_neighbours) for i in range(popu_size)]
    evas=Evaluate.evaluate(population,target)

    reference=Init.init_reference_point(evas,objectives)

    gbips=[Evaluate.gbip(weights[i],evas[i],reference, 2, objectives) for i in range(popu_size)]

    offsprings=Crossover.crossover(population,neighbours,max_complexity)
    offspring_evas=Evaluate.evaluate(offsprings,target)

    EP=[]
    EP_eva=[]

    iter_count=0
    while(iter_count<iter_limit):
        iter_count+=1
        print(f"\t GEN: {iter_count}")

        reference=Evaluate.update_ref(offspring_evas,reference,objectives)
        EP, EP_eva=Selection.update_EP(offsprings,offspring_evas,EP,EP_eva,objectives)
        population, evas, gbips=Selection.selection(weights,population,evas,offsprings,offspring_evas,neighbours,gbips,objectives,reference, replace_limit)

        offsprings=Crossover.crossover(population,neighbours,max_complexity)
        offspring_evas=Evaluate.evaluate(offsprings,target)

        Visual.distribution(EP,target)
        print(reference)
        print(population[0])
        print(offsprings[0])
        temp=""
        for key, value in evas[0].items():
            temp+=f"{key}: {round(value,2)} "
        print(temp)

    Visual.fronts(EP, EP_eva)
    for i in range(len(EP)):
        print(EP[i])
        print(EP_eva[i])

run()




