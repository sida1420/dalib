
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

    vis=Visual.MapVisualizer(map)
    print(chunks)
    popuSize=20
    population=[Init.init(width,height,start,goal) for i in range(popuSize)]
    eva=Evaluate.evaluate(map,population)

    limit=50
    gensCount=0
    freq=1


    while gensCount<limit:
        gensCount+=1
        print(f"GEN {gensCount}")
        start_time = time.perf_counter()
        if gensCount%freq==0:
            if  gensCount>50:
                freq=20
            elif gensCount>20:
                freq=10
            elif gensCount>10:
                freq=5
            elif gensCount>5:
                freq=2
            

            vis.clear_paths()
            vis.draw_paths(population,True)
            vis.save(f"Generations/gen{gensCount}.svg")

        # print("a")
        normalize_time=time.perf_counter()

        population, eva, level, crowding=Selection.selection(population,eva,popuSize)
        if gensCount==limit:
            Visual.fronts(population,level,eva)
        selection_time=time.perf_counter()
        # print("b")

        offsprings=Crossover.crossover(map,population,eva,popuSize, level, crowding)
        crossover_time=time.perf_counter()
        # print("c")
        eva+=Evaluate.evaluate(map, offsprings)
        evaluate_time=time.perf_counter()
        population+=offsprings
        print(normalize_time-start_time,selection_time-normalize_time, crossover_time-selection_time, evaluate_time-crossover_time)
        print(eva[0])

    

run()
