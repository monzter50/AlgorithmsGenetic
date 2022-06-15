import numpy as np
import random
import operator
import math as mt
import matplotlib.pyplot as plt
import pandas as pd


population_sizeX = 20
population_sizeY = 100
population_set = []
distances = []
#city_list = []
city_list = [[1,3],[2,7],[2,5],[4,8],[4,7],[4,4],[4,2],[5,3],[6,6],[6,1],[7,8],[8,7],[8,2],[9,3],[10,7],[11,6],[11,4],[11,1],[12,7],[13,5]]
cities_dict = {}


def distance(a,b):
    return mt.sqrt((((b[0]-a[0])**2)+((b[1]-a[1])**2)))

# for i in range(0,20):
#     x = int(random.random() * 10)
#     y = int(random.random() * 10)
#     city_list.append([x,y])
#     cities_dict[i]=[x,y]
#     print('Ciudad ' + str(i),'X=' + str(x),'Y=' + str(y))

def fitness(route,city_list):
    #Calculate the fitness and return it.
    score=0
    for i in range(1,len(route)):
        x_dis=route[i-1]
        y_dis=route[i]
        score = score + distance(x_dis,y_dis)
    return score
    
#Initialize populations
def initial_population(population_size, population):
    population_set = []
    for _ in range(population_size):
        city = random.sample(population,len(population))
        population_set.append(city)
    return population_set

#Function that will be used to make the list of parent route
def selection(population_ranked, elite_size):
    selection_results = []
    df = pd.DataFrame(np.array(population_ranked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    
    for i in range(0, elite_size):
        selection_results.append(population_ranked[i][0])
    for i in range(0, len(population_ranked) - elite_size):
        pick = 100*random.random()
        for i in range(0, len(population_ranked)):
            if pick <= df.iat[i,3]:
                selection_results.append(population_ranked[i][0])
                break
    return selection_results

#This function takes a population and orders it in descending order using the fitness of each individual
def sorted_routes(population,city_list):
    fitness_dict = {}
    for i in range(0,len(population)):
        fitness_dict[i] = fitness(population[i],city_list)
    sorted_results=sorted(fitness_dict.items(), key =operator.itemgetter(1), reverse = True)
    return sorted_results

#Create function to mutate a single route
def mutate(individual, mutation_rate):
    for swapped in range(len(individual)):
        if(random.random() < mutation_rate):
            swap_with = int(random.random() * len(individual))
            
            city1 = individual[swapped]
            city2 = individual[swap_with]
            
            individual[swapped] = city2
            individual[swap_with] = city1
    return individual


#Create function to run mutation over entire population
def mutate_population(population, mutation_rate):
    mutated_pop = []
    
    for ind in range(0, len(population)):
        mutated_ind = mutate(population[ind], mutation_rate)
        mutated_pop.append(mutated_ind)
    return mutated_pop

#Create a crossover function for two parents to create one child
def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        

    childP2 = [item for item in parent2 if item not in childP1]
 
    child = childP1 + childP2

    return child

#Create function to run crossover over full mating pool
def breed_population(matingpool, elite_size):
    children = []
    length = len(matingpool) - elite_size
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,elite_size):
        children.append(matingpool[i])
    
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children

#Create mating pool
def mating_pool(population, selection_results):
    matingpool = []
    for i in range(0, len(selection_results)):
        index = selection_results[i]
        matingpool.append(population[index])
    return matingpool

def next_generation(current_gen, elite_size, mutationRate,city_list):
    popRanked = sorted_routes(current_gen,city_list)
    selectionResults = selection(popRanked, elite_size)
    matingpool = mating_pool(current_gen, selectionResults)
    children = breed_population(matingpool, elite_size)
    next_generation = mutate_population(children, mutationRate)
    return next_generation


def genetic_algoritm(population,population_size,elite_size,mutation,generations):
    population = initial_population(population_size,population)
    progress = [1 / sorted_routes(population,city_list)[0][1]]
    print("Distancia Inicial: " + str(progress[0]))

    import time
    
    first_generation = True
    for i in range(1, generations+1):
        pop = next_generation(population, elite_size, mutation,city_list)
        progress.append(1 / sorted_routes(population,city_list)[0][1])
        if i%1==0:
            print('Generation '+str(i),"Distance: ",progress[i])
            plt.figure(1)
            plt.plot(progress)
            plt.ylabel('Distance') 
            plt.xlabel('Generation')
            plt.title('Better Distances vs Generation')
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)
            plt.clf()

            best_route_index = sorted_routes(population,city_list)[0][0]
            best_route = pop[best_route_index]
            x=[]
            y=[]
            print(best_route)
            for i in best_route:
                x.append(i[0])
                y.append(i[1])
                
            x.append(best_route[0][0])
            y.append(best_route[0][1])     
            
            plt.figure(2)
            plt.plot(x, y, '--o')
            plt.xlabel('X')
            plt.ylabel('Y')
            ax=plt.gca()
            plt.title('Rutas Vs Ciudades')
            bbox_props = dict(boxstyle="circle,pad=0.3", fc='C0', ec="black", lw=0.5)
            for i in range(1,len(city_list)+1):
                ax.text(city_list[i-1][0], city_list[i-1][1], str(i), ha="center", va="center",size=8,bbox=bbox_props)
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)
            plt.clf()

            if first_generation:
                time.sleep(30)
                first_generation = False
          
    return best_route

best_route=genetic_algoritm(population=city_list, population_size=100, elite_size=20, mutation=0.01, generations=300)

x=[]
y=[]

for i in best_route:
    x.append(i[0])
    y.append(i[1])

x.append(best_route[0][0])
y.append(best_route[0][1])
plt.figure(3)
plt.plot(x, y, '--o',color='green')
plt.xlabel('X')
plt.ylabel('Y')
ax=plt.gca()
plt.title('Ruta Final')
bbox_props = dict(boxstyle="circle,pad=0.3", fc='green', ec="black", lw=0.5)

for i in range(1,len(city_list)+1):
    ax.text(city_list[i-1][0], cityList[i-1][1], str(i), ha="center", va="center", size=8, bbox=bbox_props)
plt.tight_layout()
plt.show()
print(best_route)