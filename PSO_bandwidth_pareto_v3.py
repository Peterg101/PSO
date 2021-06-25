import numpy as np
import pandas as pd
import random
import csv
from matplotlib import pyplot as plt
from PSO_v2 import PSO as pso


class PSO_bandwidth():

    def __init__(self):
        self.optimal_data = pd.read_csv("Best_Data_PSO.csv", delimiter=",", encoding ="ISO-8859-1").to_numpy()
        self.number_of_frequencies_for_bw = 3
        self.minimum_frequency = 8.3e9
        self.num_data_points = 4

        #Loop Variables
        self.num_epochs = 1000
        self.num_generations = 20

        #random step integers
        self.plus_minus = [-1,0,1]
        self.random_integer_best_ever = random.randint(0,2)
        self.random_integer_current_best = random.randint(0,(pso.num_materials/2))
        
    def select_optimal_values_and_relevant_data(self):
        #Select the optimum values for the chosen bandwidth from the database
        #Select the relevant multilayer databases for the chosen bandwidth
        self.optimal_values_array = np.zeros((self.number_of_frequencies_for_bw, self.optimal_data.shape[1]), dtype = float)
        self.relevant_data_array = np.zeros((self.number_of_frequencies_for_bw, pso.num_materials, self.num_data_points), dtype = float)
        for i in range(0, self.number_of_frequencies_for_bw):

            val = np.array(np.where(self.optimal_data == (self.minimum_frequency+i*pso.increment)))[0][0]
            self.optimal_values_array[i] = self.optimal_data[val]
            self.relevant_data_array[i] = pd.read_csv("databases/multilayer_database_"+str(int(self.minimum_frequency+(pso.increment*i)))+".csv")
           

    def generate_particles_and_baselines_pareto(self):
        self.population = np.random.choice(pso.num_materials, size = pso.population_size).reshape(pso.num_particles, pso.num_layers)
        #print(self.population)

        self.best_particles = np.full(( pso.num_particles, pso.num_layers),pso.max_number, dtype = int)
        self.best_RL_values = np.full((self.number_of_frequencies_for_bw, pso.num_particles, 1), pso.max_number, dtype = float)
        self.best_ever_particle = np.full((self.number_of_frequencies_for_bw, 1,pso.num_layers), pso.max_number, dtype = int)
        

        
        self.best_compiled_values = np.full((pso.num_particles, 1), pso.max_number, dtype = float)
        self.best_ever_compiled_value = np.full(( 1), pso.max_number, dtype = float)
        self.best_ever_RL_value = np.full((self.number_of_frequencies_for_bw, 1), pso.max_number, dtype = float)
        
    def calculate_fitness_bw(self):
             
       
        #print(self.relevant_data_array)
        self.RL_fitness_array = np.zeros((self.number_of_frequencies_for_bw, pso.num_particles, 1), dtype = float)
        self.fitness_penalty_array = np.zeros((self.number_of_frequencies_for_bw, pso.num_particles, 1), dtype = float)

        #print(RL_array)
        for f in range(0, self.number_of_frequencies_for_bw):
            

            for j in range(0, pso.num_particles):
              #Top layer
              self.first_layer = self.population[j][0]
              self.first_layer_e = self.relevant_data_array[f][self.first_layer][1]
              self.first_layer_u = self.relevant_data_array[f][self.first_layer][2]
              self.first_layer_d = self.relevant_data_array[f][self.first_layer][3]
              self.zm_first_layer = np.sqrt(self.first_layer_u/self.first_layer_e)
              z_first = self.zm_first_layer*(np.tanh((np.pi*np.sqrt(self.first_layer_e*self.first_layer_u))/pso.wl)*self.first_layer_d)
            
              z_list = [z_first]
              zm_list = [self.zm_first_layer]
            
              #Calculation of RL
              for i in range(1, pso.num_layers):
                  self.layer = self.population[j][i]
                  self.layer_e = self.relevant_data_array[f][self.layer][1]
                  self.layer_u = self.relevant_data_array[f][self.layer][2]
                  self.layer_d = self.relevant_data_array[f][self.layer][3]
                  self.zm_layer = np.sqrt(self.layer_u/self.layer_e)
                  self.z_layer_minus1 = z_list[i-1]
                  self.zm_layer_minus1 = zm_list[i-1]
                
                  self.z_layer = self.zm_layer *((self.z_layer_minus1+(self.zm_layer*np.tanh((2*np.pi*np.sqrt(self.layer_e*self.layer_u))/pso.wl)*self.layer_d)))/(self.zm_layer+(self.z_layer_minus1*np.tanh(((2*np.pi*np.sqrt(self.layer_e*self.layer_u))/pso.wl)*self.layer_d)))
                                
                  zm_list.append(self.zm_layer)
                  z_list.append(self.z_layer)
                
              z_array= np.array(z_list)
              zm_array = np.array(zm_list)

              #Final RL Calculation
              RL = 20*np.log(abs((z_array[pso.num_layers-1]-1)/(z_array[pso.num_layers-1]+1)))
              
              #Impedance Matching
              zm_array_sorted = np.sort(zm_array)
            
              if (np.array_equal(zm_array, zm_array_sorted)):
                  impedance_matcher = False
              else:
                  impedance_matcher = True
        
              impedance_penalty = (int(impedance_matcher)*pso.fitness_penalty)
              #Unique Materials 
              unique_materials = (len(np.unique(self.population[j])))
              
              unique_penalty=(int(unique_materials != pso.num_layers)*pso.fitness_penalty)
              #print(unique_penalty)
              #Better than optimal solution penalty
              optimal_solution = self.optimal_values_array[f][1]
              #print(optimal_solution)
              better_than_optimal_penalty = (int(RL <= optimal_solution)*pso.fitness_penalty)
              
              self.fitness_penalty_array[f][j] = unique_penalty+impedance_penalty+better_than_optimal_penalty

              
              
              self.RL_fitness_array[f][j] = optimal_solution-RL
        
        
        self.RL_fitness_array = self.RL_fitness_array + self.fitness_penalty_array
        
        
        self.compiled_fitness_array = abs(np.sum(self.RL_fitness_array, axis = 0))
        
        
    def particle_rankings(self):
        


        for c in range(0, pso.num_particles):
            #print(self.compiled_fitness_array[c])
            if self.compiled_fitness_array[c][0] < self.best_compiled_values[c][0]:
              self.best_compiled_values[c][0] = self.compiled_fitness_array[c][0]
              self.best_particles[c] = self.population[c]

        if self.compiled_fitness_array[np.argmin(self.compiled_fitness_array)] < self.best_ever_compiled_value:
            self.best_ever_compiled_value = self.compiled_fitness_array[np.argmin(self.compiled_fitness_array)]
            self.best_ever_particle = self.population[np.argmin(self.compiled_fitness_array)]

            for i in range(0, self.number_of_frequencies_for_bw):
                
              self.best_ever_RL_value[i] =  self.RL_fitness_array[i][np.argmin(self.compiled_fitness_array)]


            #self.best_ever_RL_value = np.full((self.number_of_frequencies_for_bw, 1), pso.max_number, dtype = float)
        else:
            pass

    
    def full_bandwidth_particle_swarm_optimiser(self):
        best_particle_combinations_array = np.zeros((self.num_epochs, pso.num_layers), dtype=int)
        best_compiled_values_array = np.zeros((self.num_epochs, 1), dtype = float)
        best_RL_values_array = np.zeros((self.num_epochs, self.number_of_frequencies_for_bw, 1), dtype = float)
        for y in range(0, self.num_epochs):
            print("Epoch "+ str(y+1))
            self.select_optimal_values_and_relevant_data()
            self.generate_particles_and_baselines_pareto()

            for z in range(0, self.num_generations):
                #print("Generation "+ str(z+1))
                self.calculate_fitness_bw()
                self.particle_rankings()
                #Special case only for the first generation
                if z == 0:

                    #Move in a random direction for the first generation
                    random_array = np.random.choice(self.plus_minus, size = (self.population.shape))
                    self.population = self.population + random_array
                    self.population[self.population >= pso.num_materials] = pso.num_materials-1
                    self.population[self.population <= -1] = 0

                    #Move towards the best solution
                    difference_array = self.population-self.best_ever_particle
                    difference_array[difference_array > 0] = 1
                    difference_array[difference_array < 0] = -1
                    difference_array = difference_array*-1

                else:
                    #calculate difference between current and best positions
                    difference_current_vs_particle_best = self.population - self.best_particles
                    difference_current_vs_particle_best[difference_current_vs_particle_best > 0] = 1
                    difference_current_vs_particle_best[difference_current_vs_particle_best <0] = -1
                    difference_current_vs_particle_best = difference_current_vs_particle_best*-1


                    #calculate difference between current positions and the best position ever achieved
                    difference_current_vs_particle_best_ever = self.population - self.best_ever_particle
                    difference_current_vs_particle_best_ever[difference_current_vs_particle_best_ever > 0] = 1
                    difference_current_vs_particle_best_ever[difference_current_vs_particle_best_ever < 0] = -1
                    difference_current_vs_particle_best_ever = difference_current_vs_particle_best_ever*-1


                    #Calculate the distance to be moved in the current direction
                    current_direction = difference_current_vs_particle_best * difference_current_vs_particle_best_ever

                    #Set boundary conditions
                    self.population[self.population >= pso.num_materials] = (pso.num_materials - 1)
                    self.population[self.population < 0] = 0

                if z == self.num_generations-1:
                    best_compiled_values_array[y] = self.best_ever_compiled_value
                    best_particle_combinations_array[y] = self.best_ever_particle
                    best_RL_values_array[y] = self.best_ever_RL_value
                    
                    
                    
                    

            
        #print(best_RL_values_array)
        #print(best_particle_combinations_array)
        #print(best_RL_values_array)
        #print(best_RL_values_array.shape)
        #print(best_particle_combinations_array)
        index_comp = np.argmin(best_compiled_values_array)
        optimal_material_combination = best_particle_combinations_array[index_comp]
        print(optimal_material_combination)

        rowsum = best_RL_values_array.sum(axis = 1)
        rowsum_result = np.array(np.where(rowsum < 100))[0]
        #print(rowsum_result)
        #print(best_RL_values_array)
        #print(best_RL_values_array.shape
        
                   

        
        
        if self.number_of_frequencies_for_bw == 2:
            x = []
            y = []

            for i in range(0, len(rowsum_result)):
                x.append(best_RL_values_array[rowsum_result[i]][0])
                y.append(best_RL_values_array[rowsum_result[i]][1])

            
            
            plt.scatter(x,y)
            plt.show()
        
        
        elif self.number_of_frequencies_for_bw == 3:
            x = []
            y = []
            z = []
            for i in range(0, len(rowsum_result)):
                x.append(best_RL_values_array[rowsum_result[i]][0])
                y.append(best_RL_values_array[rowsum_result[i]][1])
                z.append(best_RL_values_array[rowsum_result[i]][2])
            

            ax = plt.axes(projection='3d')
            ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5)

            plt.show()
        
      #elif self.number_of_frequencies_for_bw == 1 or self.number_of_frequencies_for_bw >3:
          
        
        
        
        
        
        




PSObw = PSO_bandwidth()
"""
PSObw.select_optimal_values_and_relevant_data()
PSObw.generate_particles_and_baselines_pareto()
PSObw.calculate_fitness_bw()
PSObw.particle_rankings()
"""
PSObw.full_bandwidth_particle_swarm_optimiser()
