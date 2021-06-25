import numpy as np
import pandas as pd
import csv
import random


class PSO():
    
    def __init__(self):
        #Important Constants
        self.num_layers = 3
        self.num_particles = 20
        self.num_materials = 16
        self.population_size = self.num_layers*self.num_particles
        self.pi = np.pi

        #Frequency related terms
        self.f = 8e9
        self.c = 3e8
        self.wl = self.c/self.f
        self.num_freqs = 41
        self.increment = 0.1e9
        
        #Penalties
        self.fitness_penalty = 1000
        self.max_number = self.fitness_penalty*10

    def generate_particles_and_baselines(self):
        #Generate Population
        self.population = np.random.choice(self.num_materials, size = self.population_size).reshape(self.num_particles, self.num_layers)
        #print(self.population)
        self.best_particles = np.full((self.num_particles, self.num_layers),self.max_number, dtype = int)
        self.best_RL_values = np.full((self.num_particles, 1), self.max_number, dtype = float)
        self.best_ever_particle = np.full((1,self.num_layers), self.max_number, dtype = int)
        self.best_ever_RL_value = np.array([self.max_number], dtype = float)
           
        
    def calculate_fitness(self):
        
        #Constants
        #filename referenced in PSO_v2_full to enable switching for the generation of optimal freqency values
        data = pd.read_csv(self.filename, delimiter = ",", encoding = "ISO-8859-1").to_numpy()
        #RL_list = []
        self.RL_fitness_array = np.zeros((self.num_particles, 1), dtype = float)
        self.fitness_penalty_array = np.zeros((self.num_particles, 1), dtype = float)
        for j in range(0, self.num_particles):
            #Top layer
            self.first_layer = self.population[j][0]
            self.first_layer_e = data[self.first_layer][1]
            self.first_layer_u = data[self.first_layer][2]
            self.first_layer_d = data[self.first_layer][3]
            self.zm_first_layer = np.sqrt(self.first_layer_u/self.first_layer_e)
            z_first = self.zm_first_layer*(np.tanh((self.pi*np.sqrt(self.first_layer_e*self.first_layer_u))/self.wl)*self.first_layer_d)
            
            z_list = [z_first]
            zm_list = [self.zm_first_layer]
            
            #Calculation of RL
            for i in range(1, self.num_layers):
                self.layer = self.population[j][i]
                self.layer_e = data[self.layer][1]
                self.layer_u = data[self.layer][2]
                self.layer_d = data[self.layer][3]
                self.zm_layer = np.sqrt(self.layer_u/self.layer_e)
                self.z_layer_minus1 = z_list[i-1]
                self.zm_layer_minus1 = zm_list[i-1]
                
                self.z_layer = self.zm_layer *((self.z_layer_minus1+(self.zm_layer*np.tanh((2*np.pi*np.sqrt(self.layer_e*self.layer_u))/self.wl)*self.layer_d)))/(self.zm_layer+(self.z_layer_minus1*np.tanh(((2*np.pi*np.sqrt(self.layer_e*self.layer_u))/self.wl)*self.layer_d)))
                                
                zm_list.append(self.zm_layer)
                z_list.append(self.z_layer)
                
            z_array= np.array(z_list)
            zm_array = np.array(zm_list)

            #Final RL Calculation
            RL = 20*np.log(abs((z_array[self.num_layers-1]-1)/(z_array[self.num_layers-1]+1)))
            #print(RL)
            #Impedance Matching
            zm_array_sorted = np.sort(zm_array)
        
            if (np.array_equal(zm_array, zm_array_sorted)):
                impedance_matcher = False
            else:
                impedance_matcher = True
        
            impedance_penalty = (int(impedance_matcher)*self.fitness_penalty)
            #Unique Materials 
            unique_materials = (len(np.unique(self.population[j])))
              
            unique_penalty=(int(unique_materials != self.num_layers)*self.fitness_penalty)

            self.RL_fitness_array[j] = RL + unique_penalty+impedance_penalty
        #print(self.RL_fitness_array)
            
       

    def particle_rankings(self):
        #print(self.RL_fitness_array)

        for c in range(0, self.num_particles):
            if self.RL_fitness_array[c][0] < self.best_RL_values[c][0]:
                self.best_RL_values[c][0] = self.RL_fitness_array[c][0]
                self.best_particles[c] = self.population[c]
       
            #print(self.RL_fitness_array[c][0])

        if self.RL_fitness_array[np.argmin(self.RL_fitness_array)] < self.best_ever_RL_value:
            self.best_ever_RL_value = self.RL_fitness_array[np.argmin(self.RL_fitness_array)]
            self.best_ever_particle[0] = self.population[np.argmin(self.RL_fitness_array)]
        else:
            pass

       #best particles gives the best ever combination for each particle
       #best RL values gives the best ever RL value for each particle

       #best ever particle gives the best particle that has ever been discovered
       #best ever RL value gives the RL value of the best particle that has ever been discovered
 
                  
PSO = PSO()
"""
PSO.generate_particles_and_baselines()
PSO.calculate_fitness()
PSO.particle_rankings()
"""  
