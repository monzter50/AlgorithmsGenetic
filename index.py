import numpy as np
import random, operator, pandas as pd
import math as mt
import matplotlib.pyplot as plt


population_sizeX = 20
population_sizeY = 100
distances = []
#city_list = []
initial_population_list = [[1,3],[2,5],[2,7],[4,8],[4,7],[4,4],[4,2],[5,3],[6,6],[6,1],[7,8],[8,7],[8,2],[9,3],[10,7],[11,6],[11,4],[11,1],[12,7],[13,5]]
city_list = []





class City:
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def distance(self,a):
    x = abs(self.x-a.x)
    y = abs(self.y-a.y)
    return np.sqrt((x**2)+(y**2))
  
  def __repr__(self):
        return "(" + str(self.x)+ "," + str(self.y) +")"

class Fitness:
  def __init__(self,route):

    self.route = route
    self.distance = 0
    self.fitness_num = 0.0

  def route_distance(self):
    if self.distance == 0:
      path_distance = 0
      for i in range(0,len(self.route)):
        from_city = self.route[i]
        to_city = None
        if i + 1 < len(self.route):
          to_city = self.route[i + 1]
        else:
          to_city = self.route[0]

        path_distance += from_city.distance(to_city)

      self.distance = path_distance

      return self.distance

  def route_fitness(self):
    if self.fitness_num == 0:
      self.fitness_num = 1 / float(self.route_distance())
    return self.fitness_num




#Initialize populations
def initial_population(population_size, population):
    population_set = []
    for _ in range(0,population_size):
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
def sorted_routes(population):
    fitness_dict = {}
    for i in range(0,len(population)):
        fitness_dict[i] = Fitness(population[i]).route_fitness()
    sorted_results=sorted(fitness_dict.items(), key = operator.itemgetter(1), reverse = True)
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

def next_generation(current_gen, elite_size, mutationRate):
    pop_ranked = sorted_routes(current_gen)
    selection_results = selection(pop_ranked, elite_size)
    matingpool = mating_pool(current_gen, selection_results)
    children = breed_population(matingpool, elite_size)
    next_generation = mutate_population(children, mutationRate)
    return next_generation


def genetic_algoritm(population,population_size,elite_size,mutation,generations):
    _population = initial_population(population_size,population)
    progress = [1 / sorted_routes(_population)[0][1]]
    print("Distancia Inicial: " + str(progress[0]))

    import time

    first_generation = True
    
    for i in range(1, generations+1):
        _population = next_generation(_population, elite_size, mutation)
        progress.append(1 / sorted_routes(_population)[0][1])
        if i%1==0:
            print('Generation '+str(i),"Distance: ",progress[i])
            plt.subplot(221)
            plt.plot(progress)
            plt.ylabel('Distance')
            plt.xlabel('Generation')
            plt.title('Better Distances vs Generation')
            plt.tight_layout()
            # plt.show(block=False)
            # plt.pause(0.1)
            # plt.clf()

            best_route_index = sorted_routes(_population)[0][0]
            best_route = _population[best_route_index]
            x=[]
            y=[]
            print(best_route)
            for i in best_route:
                x.append(i.x)
                y.append(i.y)

            x.append(best_route[0].x)
            y.append(best_route[0].y)

            plt.subplot(222)
            plt.plot(x, y, '--o')
            plt.xlabel('X')
            plt.ylabel('Y')
            ax=plt.gca()
            plt.title('Rutas Vs Ciudades')
            bbox_props = dict(boxstyle="circle,pad=0.3", fc='C0', ec="black", lw=0.5)
            for i in range(1,len(city_list)+1):
                ax.text(city_list[i-1].x, city_list[i-1].y, str(i), ha="center", va="center",size=8,bbox=bbox_props)
            plt.tight_layout()
            plt.pause(0.1)
            plt.clf()
            plt.show(block=False)

            if first_generation:
                time.sleep(1)
                first_generation = False

    return (best_route, progress)


################################################################################################

for i in initial_population_list:
  city_list.append(City(x=i[0],y=i[1]))

best_route, progress = genetic_algoritm(population=city_list, population_size=200, elite_size=20, mutation=0.01, generations=100)

x=[]
y=[]
plt.subplot(221)
plt.plot(progress)
plt.ylabel('Distance')
plt.xlabel('Generation')
plt.title('Better Distances vs Generation')
plt.tight_layout()
for i in best_route:
  x.append(i.x)
  y.append(i.y)

x.append(best_route[0].x)
y.append(best_route[0].y)
plt.subplot(222)
plt.plot(x, y, '--o',color='green')
plt.xlabel('X')
plt.ylabel('Y')
ax=plt.gca()
plt.title('Ruta Final')
bbox_props = dict(boxstyle="circle,pad=0.3", fc='green', ec="black", lw=0.5)

for i in range(1,len(city_list)+1):
  ax.text(city_list[i-1].x, city_list[i-1].y, str(i), ha="center", va="center", size=8, bbox=bbox_props)

plt.tight_layout()
plt.show()
print(best_route)