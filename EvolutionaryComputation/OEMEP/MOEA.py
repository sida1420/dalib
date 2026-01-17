
import Init
import Visual
import Selection
import Crossover
import pickle
import time
import Evaluate
import Chunking
def run():


    map={}
    with open("EvolutionaryComputation/OEMEP/map.plk","rb") as file:
        map=pickle.load(file)
    width, height,obstacles, sensors, chunks, start, goal = map.values()
    print(chunks)
    popuSize=20
    population=[Init.init(width,height,start,goal) for i in range(popuSize)]
    eva=Evaluate.evaluate(map,population)

    limit=20
    gensCount=0

    while gensCount<limit:
        gensCount+=1
        print(f"GEN {gensCount}")
        start_time = time.perf_counter()
        Visual.mapPaths(map,population,show=False,save=f"Generations/gen{gensCount}.svg")
        print("a")
        normalize_time=time.perf_counter()

        population, eva, level, crowding=Selection.selection(population,eva,popuSize)
        selection_time=time.perf_counter()
        print("b")

        offsprings=Crossover.crossover(map,population,eva,popuSize, level, crowding)
        crossover_time=time.perf_counter()
        print("c")
        eva+=Evaluate.evaluate(map, offsprings)
        evaluate_time=time.perf_counter()
        population+=offsprings
        print(normalize_time-start_time,selection_time-normalize_time, crossover_time-selection_time, evaluate_time-crossover_time)
        print(eva[0])

    
run()
