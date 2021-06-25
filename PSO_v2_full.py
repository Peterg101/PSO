import numpy as np
import pandas as pd
import random
import csv
from PSO_v2 import PSO as pso



class PSO_full():

    def __init__(self):
        self.num_epochs = 1000
        self.num_generations = 20
        

        #random step integers
        self.plus_minus = [-1,0,1]

        self.random_integer_current_best = random.randint(0, 3) #c1

        self.random_integer_best_ever = random.randint(0,3) #c2
        

    def particle_swarm_optimiser(self):
        
        frequencies = np.arange(start= pso.f, stop= pso.f+(pso.increment*(pso.num_freqs)), step = pso.increment).reshape(pso.num_freqs,1)
        best_particle_combinations_frequency = np.zeros((pso.num_freqs, pso.num_layers), dtype = int)
        best_particle_RL_values = np.zeros((pso.num_freqs, 1), dtype = float)

        for x in range (0, pso.num_freqs):
          print("Frequency " + str(frequencies[x][0]) +"Hz")
          best_particle_combinations_array = np.zeros((self.num_epochs, pso.num_layers), dtype=int)
          best_RL_values_array = np.zeros((self.num_epochs, 1), dtype = float)
          for y in range(0, self.num_epochs):
            pso.generate_particles_and_baselines()
            pso.filename = "databases/multilayer_database_"+str(int(pso.f+(x*pso.increment)))+".csv"
            for z in range(0, self.num_generations):
              pso.calculate_fitness()
              pso.particle_rankings()
              #Special case only for the first generation
              if z == 0:
              
                  #move in a random direction for the first generation
                  random_array = np.random.choice(self.plus_minus, size = (pso.population.shape))
                  pso.population = pso.population + random_array
                  pso.population[pso.population >= pso.num_materials] = pso.num_materials-1
                  pso.population[pso.population <= -1] = 0
    
                  #move towards the best solution
                  difference_array = pso.population-pso.best_ever_particle
                  difference_array[difference_array > 0] = 1
                  difference_array[difference_array < 0] = -1
                  difference_array = difference_array*-1
              
                
              else:

                  #calculate difference between current positions and best positions
                  difference_current_vs_particle_best = pso.population - pso.best_particles
                  difference_current_vs_particle_best[difference_current_vs_particle_best > 0] = 1
                  difference_current_vs_particle_best[difference_current_vs_particle_best <0] = -1
                  difference_current_vs_particle_best = difference_current_vs_particle_best*-1
                  #print(difference_current_vs_particle_best)
                  #calculate difference between current positions and the best position ever achieved
                  difference_current_vs_particle_best_ever = pso.population - pso.best_ever_particle
                  difference_current_vs_particle_best_ever[difference_current_vs_particle_best_ever > 0] = 1
                  difference_current_vs_particle_best_ever[difference_current_vs_particle_best_ever < 0] = -1
                  difference_current_vs_particle_best_ever = difference_current_vs_particle_best_ever*-1
                  #print(difference_current_vs_particle_best_ever)
                  #calculate the distance to be moved in the currennt direction
                  current_direction = difference_current_vs_particle_best * difference_current_vs_particle_best_ever
              
                  #move the particles
                  pso.population = pso.population + current_direction + (difference_current_vs_particle_best * self.random_integer_current_best)  + (difference_current_vs_particle_best_ever * self.random_integer_best_ever)

                  #set boundary conditions
                  pso.population[pso.population >= pso.num_materials] = (pso.num_materials - 1)
                  pso.population[pso.population < 0] = 0

              if z == self.num_generations-1:
                  best_RL_values_array[y] = pso.best_ever_RL_value
                  best_particle_combinations_array[y] = pso.best_ever_particle



          index = np.argmin(best_RL_values_array)
          best_particle_RL_values[x] = best_RL_values_array[index]
          best_particle_combinations_frequency[x] = best_particle_combinations_array[index]

        print(frequencies.shape)
        print(best_particle_RL_values.shape)
        print(best_particle_combinations_frequency.shape)

        final_data_array = np.concatenate((frequencies, best_particle_RL_values, best_particle_combinations_frequency), axis = 1)
        columns = ["Frequency (Hz)", "RL Value (dB)", "Material 1", "Material 2", "Material 3"]

        best_DataFrame = pd.DataFrame(final_data_array, columns = ["Frequency (Hz)", "RL Value (dB)", "Material 1", "Material 2", "Material 3"])
        best_DataFrame.to_csv("Best_Data_PSO.csv", index = None)


        

        


PSO_f = PSO_full()

PSO_f.particle_swarm_optimiser()
