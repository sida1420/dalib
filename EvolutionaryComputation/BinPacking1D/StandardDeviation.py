
def Average(fitnesses):
    return sum(fitnesses)/len(fitnesses)

def StandardDeviation(fitnesses):
    aver=Average(fitnesses)

    sd=0
    for fitness in fitnesses:
        sd+=(fitness-aver)**2

    sd=(sd/len(fitnesses))**0.5
    return sd