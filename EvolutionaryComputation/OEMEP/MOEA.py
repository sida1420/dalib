
import Init
import Visual
import Selection
import Crossover
import pickle
import Evaluate
def run():


    map={}
    with open("EvolutionaryComputation/OEMEP/map.plk","rb") as file:
        map=pickle.load(file)
    width, height,obstacles, sensors, start, goal = map.values()

    
    popuSize=10
    population=[Init.init(width,height,start,goal) for i in range(popuSize)]
    eva=[Evaluate.evaluate(map,ind) for ind in population]

    limit=10

    while limit:
        limit-=1

        population, eva, level, crowding=Selection.selection(population,eva,popuSize)
        Visual.mapPaths(map,population)

        offsprings=Crossover.crossover(map,population,eva,popuSize, level, crowding)
        eva+=[Evaluate.evaluate(map, ind) for ind in offsprings]
        population+=offsprings
        print(eva[0])

    
run()
