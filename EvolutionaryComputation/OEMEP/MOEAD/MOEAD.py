
import Init
import Visual
import Selection
import Crossover
import pickle
import time
import Evaluate
import Chunking

objectives=("exposure","distance")
def run():


    map={}
    with open("EvolutionaryComputation/OEMEP/NSGAII/map.plk","rb") as file:
        map=pickle.load(file)
    width, height,obstacles, sensors, chunks, start, goal = map.values()

    vis=Visual.MapVisualizer(map)
    print(chunks)
    numDivisions=39
    numObjectives=2
    numNeighbours=5

    weightVectors=Init.initWeightVectors(numDivisions,numObjectives)
    popuSize=len(weightVectors)
    population=[Init.init(width,height,start,goal) for i in range(popuSize)]
    neighbours=[Init.whoAmINeighbour(i,weightVectors,numNeighbours) for i in range(popuSize)]
    
    eva=Evaluate.evaluate(map,population)
    referencePoint=Init.initReferencePoint(eva,objectives)
    gbips=[Evaluate.gbip(weightVectors[i],eva[i],referencePoint,0.5,objectives) for i in range(popuSize)]

    offsprings=Crossover.crossover(map,population,neighbours,popuSize)
    offspringEva=Evaluate.evaluate(map,offsprings)
    limit=50
    gensCount=0
    freq=1

    EP=[]
    EPEva=[]

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

        normalize_time=time.perf_counter()
        referencePoint=Evaluate.updateRef(offspringEva,referencePoint,objectives)
        EP, EPEva=Selection.updateEP(offsprings,offspringEva, EP, EPEva, objectives)
        population, eva, gbips=Selection.selection(weightVectors,population,eva,offsprings,offspringEva,neighbours,gbips,popuSize,objectives,referencePoint, 3)
        if gensCount==limit:
            Visual.fronts(EP, EPEva)
        selection_time=time.perf_counter()
        offsprings=Crossover.crossover(map,population,neighbours,popuSize)
        crossover_time=time.perf_counter()
        offspringEva=Evaluate.evaluate(map, offsprings)
        evaluate_time=time.perf_counter()
        print(normalize_time-start_time,selection_time-normalize_time, crossover_time-selection_time, evaluate_time-crossover_time)
        print(eva[0])

    

run()
