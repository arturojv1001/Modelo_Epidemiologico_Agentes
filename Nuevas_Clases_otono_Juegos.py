import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.animation as animation
from tqdm import tqdm
import os, shutil, glob
# from scipy.stats import poisson
# from scipy.stats import moyal
import statistics

# ==================================================================================
#                               Funciones
# ==================================================================================


class Modelo_Epidemia:
    
    def __init__(self,N,steps):  #initialize self, with global parameters
        self.N = N           # number of nodes
        self.steps = steps   # number of steps of the epidemic
        return

    def list_of_state(self,G,state_of_nodes):
        '''
        This Function looks for the infected nodes in a graph by searching for those nodes with state = "I" 
            Input:
                        G               - (networkx graph)
                        state_of_nodes  - (string) 'S','I','R' or 'D' refering to the state of a set of nodes of interest 
            Output:
                        (list) array of nodes with the certain state of interest ('S','I','R' or 'D')
        '''
        
        nodes_dict =  dict(G.nodes(data='state'))  # we obtain the dictionary with keys as nodes 
        # then we run a loop over the keys of the dictionary to select the nodes with the respective state introduced in the functions input
        return [nodo  for (nodo,value) in nodes_dict.items() if value == state_of_nodes] 
    
    
    
    def change_state(self,Nodes_of_Certain_State,G,new_state):
        '''
        This Function changes the state of the nodes of a certain group of nodes introduced in the input 
            Input:
                    Nodes_of_Certain_State - (list) List of nodes that will change state
                    G                      - (networkx graph) Graph in which the nodes will change their state
                    new_state              - (string) the state that the nodes will change into ('S','I','R' or 'D')
            Output:
                    G                      - (networkx graph) Graph with the nodes states updated 
        '''
        
        for nodo in Nodes_of_Certain_State:  # we make a loop through the list of the nodes that want to be changed
            G.nodes[nodo]['state'] = new_state  # the states are changed to the input variable "new_state"
        
        return G
    
    
    def change_state_CoD(self,Nodes_of_Certain_State,G,new_state):
        '''
        This Function changes the state of Cooperator or Detractor of the nodes of a certain group of nodes introduced in the input 
            Input:
                    Nodes_of_Certain_State - (list) List of nodes that will change 'Cooperator'/'Detractor' state
                    G                      - (networkx graph) Graph in which the nodes will change their 'Cooperator'/'Detractor' state
                    new_state              - (string) the state that the nodes will change into ('C' or 'D')
            Output:
                    G                      - (networkx graph) Graph with the node´s 'Cooperator'/'Detractor' states updated 
        '''
        
        for nodo in Nodes_of_Certain_State:  # we make a loop through the list of the nodes that want to be changed
            G.nodes[nodo]['CoD'] = new_state  # the states are changed to the input variable "new_state"
        
        return G
    
    
    def intersection(self,lst1, lst2):
        '''
        This function gives the intersection between two lists
        
        Input: 
                 lst1 - (list) fisrt list
                 lst2 - (list) second list
        Output:
                 lst3 - (list) intersection between fisrt and second lists
        '''
        
        lst3 = []           # we initialize the list where both lists will intersect
        for value in lst1:      # loop through the first list
            if value in lst2:   # if an element of the first list is also in the second list
                lst3.append(value)  # that element is added to the list of the intersection
        return lst3   # it returns the list of all elements that belong to both lists
    
    
    def Binomial_Election(self,List_of_Nodes,probability_state):
        '''
        This function selects a sample from a list of nodes by using a binomial distribution with a given probability
        
        Input: 
                 List_of_Nodes      - (list) list of nodes where the sample will be selected from
                 probability_state  - (float) probability for the binomial distribution
        Output:
                 State_Sample       - (list) sample of nodes selected from the binomial process
        '''
    # binomial with parameters n=num_sample and p=probability_state (introduced in the input)
        num_sample = np.random.binomial(len(List_of_Nodes), probability_state)
    # we select a random sample the size of 'num_sample' from the list of nodes     
        State_Sample = random.sample(List_of_Nodes,num_sample)
        return State_Sample
    
    
    
    def n_steps_before_state(self,Graph_Array,state_of_nodes,n_steps):
        '''
        Function to search for the list of nodes that had a certain state n steps in the past
        
        Posible states: 'S' (susceptible), 'I' (infected), 'R' (recovered), 'D' (deceased)
    
        Input:
                    Graph_Array       - (list) Array that contains each graph updated every period of time in the epidemic
                    state_of_nodes    - (string) the state of the nodes that are being searched for 
                    n_steps           - (int) Number of steps into the past where the searching will occur
    
        Output:
                    searched_nodes    - (list) the list of nodes that had a certain state n steps in the past
    
        '''
        
        total_lenght = len(Graph_Array)     # obtain the lenght of the array of graphs from the input
    # obtain the graph of the epidemic "n_steps" in the past from the current period
        G = Graph_Array[total_lenght-1 - n_steps]   
    # once the graph is obtained, it searches for the list of nodes with a certain state in that graph
        searched_nodes = self.list_of_state(G,state_of_nodes)  # the state that it´s being searched for is defined in the input of the function
        
        return searched_nodes
    
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
###################################################################################################################################################################    
    
    def Initial_Graph_atribute(self,N_infected,graph_name,detractor_prob,W_max):
        '''
        Function to define the initial graph in which it randomly selects which nodes will start as infected or 
        suscpetible with the next graph atribute
        
        'state' = String 'S' (susceptible) or 'I' (infected)
    
        Input:
                    N                - (int) Number of nodes (global variable)
                    N_infected       - (int) Number of initial infected nodes
                    graph_name       - (string) Name of the type of random graph that will be selected
                    detractor_prob   - (float) The probability that an individual is a detractor 
                    W_max            - (float) The maximum weight an edge can have 
    
        Output:
                    G              - (networkx graph) graph with inicialized atributes and edge weights 
    
        '''
##########################################################################################################################
        # section to select the type of graph that will be used for the simulation
        if graph_name == 'ER'   : G = nx.erdos_renyi_graph(self.N,0.6)
        if graph_name == 'NWS'  : G = nx.newman_watts_strogatz_graph(self.N,4,0.5)
        if graph_name == 'BA'   : G = nx.barabasi_albert_graph(self.N,2)
    
        # Once the graph is created, we define the next atributes for each node:
        #         'state'  - which defines the state of the node in the epidemic (Susceptible,Infected,Recovered,Deceased)
        #         'days_I' - amount of days that a node has been infected
        #         'days_R' - amount of days that a node has been recovered
        #         'CoD'    - state that defines wether a node is a detractor or cooperator (for the Prisioner´s Dilemma Game), every node is inicialized as a Coopertator and then the Detractors are selected
        # 'Vulnerabilidad' - a number that represents how vulnerable an individual is towars infection, the greater the number the more vulnerable a node is
        #      'Utiilidad' - the amount of utility (well-being) an individual has, we inicialize every node will null utility
        #        'Peligro' - the amount (numerical) of danger a node has at a given time in the simulation, also inicialized with 0
        #     'Exposicion' - the amount (numerical) of exposure a node has at a given time in the simulation, also inicialized with 0
##########################################################################################################################
        # Atribute asignation section
        for nodo in G.nodes():
            G.nodes[nodo]['state']  = 'S'       
            G.nodes[nodo]['days_I'] = 0
            G.nodes[nodo]['days_R'] = 0
            G.nodes[nodo]['CoD']    = 'C'
            G.nodes[nodo]['Vulnerabilidad'] = 1
            G.nodes[nodo]['Utilidad'] = 0
            G.nodes[nodo]['Peligro']  = 0
            G.nodes[nodo]['Exposicion'] = 0
##########################################################################################################################
        # Edge weight distribution Section    
        for u,v,w in G.edges(data=True):
            s = np.random.power(4, 1)   # we use a scale free distribution for the edge´s weights
            s = ((1-s)*(W_max)+1)       # we transform the distribution into a range of numbers between 1 and W_max
            w['weight'] = s             # we asign the number from the distribution to the weight of the edge 
##########################################################################################################################
        # Section to select the nodes that will be detractors in the simulation
        Detractores = self.Binomial_Election(G.nodes(),detractor_prob)  # we call a function to select a random sample from the graph´s nodes given a probability
        G = self.change_state_CoD(Detractores,G,'D')    # we change the 'CoD' state to 'D' for the nodes from the previous sample    
##########################################################################################################################
        # Section to select the initial infected node(s) in the graph
        
        # A random sample of size 'N_infected' is selected from the node
        seeds  = random.sample(list(G.nodes()), N_infected)
    
        # Loop through the intial infected to change their state from 'S' to 'I'
        for nodo in seeds: G.nodes[nodo]['state'] = 'I'
        # the number of days infected for the first graph are initialized
        G = self.add_or_reset_state_days(seeds, G, 'days_I','add')
    
        return G
    
    
    def add_or_reset_state_days(self,Nodes_of_Certain_State,G,I_R_days,add_or_reset):
        '''
        This Function adds(+1) or resets(=0) the number of days that a node has been in a certain state
            Input:
                    Nodes_of_Certain_State   - (list) nodes that will change state
                    G                        - (networkx graph) Graph in which the nodes will change their state
                    I_R_days                 - (string) state of the number of days that will be updated
                    add_or_reset             - (string) 'add' will add a day to the number of days and 'reset' will make it 0
            Output:
                    G                        - (networkx graph) Graph with the nodes days with a certain state updated 
        '''
        if add_or_reset == 'add':
            for nodo in Nodes_of_Certain_State:  # we make a loop through the list of the nodes that want to be changed
                G.nodes[nodo][I_R_days] = G.nodes[nodo][I_R_days] + 1  # the states are changed to the input variable "new_state"
        elif add_or_reset == 'reset':
            for nodo in Nodes_of_Certain_State:  # we make a loop through the list of the nodes that want to be reseted
                G.nodes[nodo][I_R_days] = 0  # the number of days on that certain state resets to 0
        return G


    def Count_Days_I_R(self,G,r_period,state_counted):
        '''
        This Function obtains the list of nodes that have been at least 'r_period' amount of days in a certain state}
            Input:
                    G                     - (networkx graph) Current graph of the epidemic
                    r_period              - (int) minimal number of days a node has to be in a certain state to be included in the output list
                    state_counted         - (string= 'I' or 'R') the state in which the list of nodes have been for a certain time
            
            Output:
                    List_Nodes_Days_I_R   - list of nodes that have been at least 'r_period' amount of days in a certain state
        '''
        List_Nodes_Days_I_R = []   # we initialize the list of nodes 
        atribute = 'days_'+state_counted     # create a string by adding 'days' to the state introduced in the input
        for person in self.list_of_state(G, state_counted):   # Loop through the list of Infected or Recovered in the graph
            if G.nodes[person][atribute] >= r_period:  # if the number of days of the state of interest in a node is at least r_period days
                List_Nodes_Days_I_R.append(person)     # the node is added to the list of nodes of the output
        return List_Nodes_Days_I_R
    
    
    
    def Infections_t(self,G_next,p_i,acc_inf,incubation_period):
        '''

        Function that simulates infections by looping through every infected node and picking a random sample from the 
        list of its susceptible neighbors by using a binomial process. After selecting the sample, the epidemic state 
        of the nodes in the sample are changed from 'S' (susceptible) to 'I' (infected).
            Input:
                    G_next              - (networkx graph) Current graph of the epidemic
                    p_i                 - (float) probability of infection of the epidemic
                    acc_inf             - (int) number of accumulated infected throughout the epidemic
                    incubation_period   - (int) minimal number of days a node has to be infected before it can infect other nodes
            
            Output:
                    acc_inf  (int)              - updated number of accumlated infected after adding those who get infected in the current iteration

        '''
        Infectados = self.Count_Days_I_R(G_next, incubation_period, 'I')
        for infected_node in Infectados:  # loop through the list of current infected nodes
        # we obtain the list of suceptible neighbors of each infected node by intersecting the list 
        # of susceptible nodes and the list of the neighbors of an infected node
            susceptible_neighbors = self.intersection(self.list_of_state(G_next,'S'), list(G_next.neighbors(infected_node)) )
            
            if len(susceptible_neighbors) != 0:     # if the infected node has no susceptible neighbors
                # if the infected node has at least one suscpetible neighbor 
                # first we obtain the number of neighbors that will get infected by an infected node with a binomial process
            # binomial with parameters n=number of susceptible neighbors and p=probability of infection (introduced in the input)
                #num_new_inf_neighbors = np.random.binomial(len(susceptible_neighbors), p_i)   
                
            # we select a random sample from the susceptible neighbors of the size of "num_new_inf_neighbors" 
                new_inf_neighbors = self.Binomial_Election(susceptible_neighbors,p_i)
            # the number of accumulated infected is updated by adding the number of new infected by a node
                acc_inf = acc_inf + len(new_inf_neighbors)
            # we change the state of the new infected from 'S' susceptible to 'I' infected
                self.change_state(new_inf_neighbors,G_next,'I')
        return acc_inf
    
    
    
    def Infections_Box2(self,G_next,p_i,acc_inf,incubation_period):
        '''

        Function that simulates infections by looping through every infected node and picking a sample from the 
        list of its susceptible neighbors by using a binomial process. However, the selction of the sample is not totally
        random as the probability of selecting a susceptible neighbor that will get infected is determined by the weight
        of the edge it has with the infected node.In particular this probability is defined as 
        p_ij = (weight of edge(i,j)) /sum(weights of edges between the infected node and its susceptible neighbors) 
        After selecting the sample, the epidemic state of the nodes in the sample are changed from 'S' (susceptible) to 'I' (infected).
            
            Input:
                    G_next              - (networkx graph) Current graph of the epidemic
                    p_i                 - (float) probability of infection of the epidemic
                    acc_inf             - (int) number of accumulated infected throughout the epidemic
                    incubation_period   - (int) minimal number of days a node has to be infected before it can infect other nodes
            
            Output:
                    acc_inf             - (int) updated number of accumlated infected after adding those who get infected in the current iteration

        '''
        # the list of infected nodes that have gone through the incubation period is assigned to the list 'Infectados'
        Infectados = self.Count_Days_I_R(G_next, incubation_period, 'I')
        for infected_node in Infectados:  # loop through the list of infected that are able to infect the disease
            # the list of susceptible neighbors of the infected node is obtained by intersecting the list of susceptible nodes with the list of neighbors of the infected node
            susceptible_neighbors = self.intersection(self.list_of_state(G_next,'S'), list(G_next.neighbors(infected_node)) )
            if len(susceptible_neighbors) != 0:  
                # if the infected node does have susceptible neighbors, we obtain the number of susceptible neighbors that will get infected by the infected node in the current iteration
                num_new_inf_neighbors = np.random.binomial(len(susceptible_neighbors),p_i)
                
                for iteracion in range(num_new_inf_neighbors): # every iteration of this loop a susceptible node gets infected
                    W = np.array([])  # initialize empy numpy array 
                    for nodo_sus in susceptible_neighbors:    # loop through susceptible neighbors
                        W = np.append(W,float(G_next[infected_node][nodo_sus]["weight"]))   # create a list with the weights of the edges between infected node and its susceptible neighbors
                    # a susceptible neighbor is selected to get infected with a probability proportional to its edge weight with the infected node
                    new_infected = np.random.choice(susceptible_neighbors, p=W/sum(W))   
                
                    self.change_state([new_infected],G_next,'I')  # the state of the new infected node is changed from 'S' to 'I'
                    susceptible_neighbors.remove(new_infected)    # the new infected node is removed from the list of susceptible neighbors of the current infected node
                
                
            # the number of accumulated infected is updated by adding the number of new infected by a node
                acc_inf = acc_inf + num_new_inf_neighbors
            
        return acc_inf
    
    
    def Infected_to_Recovered(self,G_next,infectious_period,p_r):
        '''
        Function to change the epidemic state of a node from 'I' (infected) to 'R' (recovered). It also resets the 
        counter of infected days of a node ('days_I') to 0. 
            Input:
                    G_next              - (networkx graph) Current graph of the epidemic
                    infectious_period   - (int) minimal number of days a node has to be infected before it can recover from the disease
                    p_r                 - (float) probability of recovering from the disease
            
            Output:
                   Updates the state of the epidemic of the nodes that recover during the current iteration of the simulation  

        '''
        # we obtain the list of nodes that have been infected for at least "rrecoverable_period" amount of days
        Possible_Recovered = self.Count_Days_I_R(G_next, infectious_period, 'I')
        
        if len(Possible_Recovered)!=0:  #if there are nodes that can recover from infection
        # we select a random sample from the list of Possible Recovered through a binomial process with probability p_r
            Recovered = self.Binomial_Election(Possible_Recovered,p_r)
        # we change the state of the nodes that recover from 'I' infected to 'R' recovered
            self.change_state(Recovered,G_next,'R')
        # we reset the atribute 'days_I' to 0 of the recovered nodes
            self.add_or_reset_state_days(Recovered, G_next, 'days_I','reset')
        return
    
    
    def Recovered_to_Susceptible(self,G_next,inmmunity_period,p_s):
        '''
        Function to change the epidemic state of a node from 'R' (recovered) to 'S' (susceptible). It also resets the 
        counter of inmmune/recovered days of a node ('days_R') to 0. 
            Input:
                    G_next              - (networkx graph) Current graph of the epidemic
                    inmmunity_period    - (int) temporary inmmunity period (minimal number of days after recovering from the disease before an indivudal can get infected again)
                    p_s                 - (float) probability of becoming susceptible to the disease while being temporarily inmmune
            
            Output:
                   Updates the state of the epidemic of the nodes that become susceptible again during the current iteration of the simulation 

        '''
        # we obtain the list of nodes that have been through the minimal inmmunity period and are now able to become susceptible again
        Possible_New_Sus = self.Count_Days_I_R(G_next, inmmunity_period, 'R')
        
        if Possible_New_Sus !=0:  # if the previous mencioned list is not empty
         # a sample from the list 'Possible_New_Sus' is obtained with a binomial process
            New_Sus = self.Binomial_Election(Possible_New_Sus,p_s) 
        # the epidemic state of the nodes from the sample 'New_Sus' is changed from 'R' to 'S'
            self.change_state(New_Sus,G_next,'S')
        # we reset the atribute 'days_R' to 0 of the nodes that lost inmmunity
            self.add_or_reset_state_days(New_Sus, G_next, 'days_R','reset')
        return
    
        
    def Update_Epidemic(self,G_array,p_i,acc_inf,infectious_period,p_r,inmmunity_period,p_s,incubation_period):
        '''
        This Function updates the epidemic state of the nodes of the graph with a stochastical process
        First it simulates the infection process of the epidemic, then it simulates the recovering process for the nodes
        that have been infected for a determined time and it finally simulates the process in which temporarily inmmune 
        nodes become susceptible again.
        
    
        Input:
                    G_array              - (list) array of graphs of the epidemic over time 
                    p_i                  - (float) probability of infection 
                    acc_inf              - (int) accumulated number of infected nodes
                    infectious_period    - (int) number of periods of time before before an infected node can recover
                    p_r                  - (float) probability of recovering when infected
                    inmmunity_period     - (int) number of periods of time before before a recovered node can become susceptible again
                    p_s                  - (float) probability of becoming susceptible to the disease while being temporarily inmmune
                    incubation_period    - (int) minimal number of days a node has to be infected before it can infect other nodes
        Output:
                    G_array              - (list) array of graphs representing the epidemic over time 
                    acc_inf              - (int) updated number of accumulated infected nodes 
        '''
        G = G_array[-1]     # select the last graph of the array 
                            # this represents the graph of the current period of time
        G_next = G.copy()           # create a copy of the current graph
        
        ###################################  CONTAGION SECTION ################################################################
        ##################################################################################################################
        acc_inf = self.Infections_Box2(G_next,p_i,acc_inf,incubation_period)
        #acc_inf = self.Infections_t(G_next,p_i,acc_inf,incubation_period)
        
        ###################################  INFECTED TO RECOVERED SECTION #############################################
        ##################################################################################################################
        self.Infected_to_Recovered(G_next,infectious_period,p_r)
        
        ###################################  RECOVERED TO SUSCEPTIBLE SECTION #############################################
        ##################################################################################################################
        self.Recovered_to_Susceptible(G_next,inmmunity_period,p_s)
        
        ################################### UPDATE COUNTER OF DAYS INFECTED/RECOVERED #############################################
        ##################################################################################################################
        
        # we add a day to the atribute 'days_I' to all the nodes currently infected at the end of the period
        G_next = self.add_or_reset_state_days(self.list_of_state(G_next,'I'), G_next, 'days_I','add')
        # we add a day to the atribute 'days_R' to all the nodes currently recovered at the end of the period
        G_next = self.add_or_reset_state_days(self.list_of_state(G_next,'R'), G_next, 'days_R','add')
        
        # the Graph is updated by changing the state of the new infected and the recovered nodes
        G_array.append(G_next)
        
        return G_array,acc_inf
    
    '''    
    def Epidemic_Evolution(self,G0,p_i,r_period,p_r,inmmunity_period,p_s,incubation_period,Vector_Utilidades,W_max):
    
        """ Function to simulate the course of the epidemic over a period of time. 
    
            Input: 
                G0        - Initial Graph
                p_i       - probability of getting infected
                r_period  - number of periods of time before before an infected node can recover
                p_r       - probability of recovering when being infected
            Output:
                Gtrack              - array in which every entry is the updated graph en each step of the epidemic
                Accumulative_count  - array of accumulative number of infected nodes over time
        """
        # Before the contagion of the disease begins:
        # Every node plays the Prisioner´s Dilemma with its neighbors and obtains a certain utility on the first iteration
        G0 = self.Dilema_del_Prisionero(G0,Vector_Utilidades,p_i,W_max)
        # We obtain the initial risk of every node and substract it from is utility
        G0 = self.Riesgo(G0,p_i,W_max)
        
        Gtrack = [G0]   # initialize the array of graphs with the initial graph(introduced in the input)
        acc_inf = len(self.list_of_state(G0,'I'))   # initialize the number of accumulated infections with the number of infected in the initial graph
        Accumulative_count = [acc_inf]      # initialize the general array of accumulated infectections over time
        for step_i in tqdm(range(1,self.steps)):
            # Simulate the epidemic and update the epidemic state of the nodes in the current iteration
            Gtrack, acc_inf = self.Update_Epidemic(Gtrack,p_i,acc_inf,r_period,p_r,inmmunity_period,p_s,incubation_period)   #Update the Graphs and number of accumulated infections in every period of time
            # Every node plays the Prisioner´s Dilemma with its neighbors and obtains a certain utility on the current iteration
            Gtrack[-1] = self.Dilema_del_Prisionero(Gtrack[-1],Vector_Utilidades,p_i,W_max)
            # We calculate the Risk of each node and subtract it from the utility of the node (as the risk is considered negative utility)
            Gtrack[-1] = self.Riesgo(Gtrack[-1],p_i,W_max)
            
            Accumulative_count.append(acc_inf)      # Add the number of accumulated infections of every period to the general array 
    
        return Gtrack, Accumulative_count
    '''
    
    
    def Dilema_del_Prisionero(self,G0,Vector_Utilidades,p_i,W_max):
        """ This function simulates the game (based no the Prisioner´s Dilemma) that nodes play with their neighbors
            to gain 'utility' in every iteration. Their profit will depend on their election of strategy. For this 
            particular game there are only 2 choices: 'C' or 'D' which stand for Cooperator or Detractor respectively. 
    
            Input: 
                G0                  - (networkx graph) Graph before playing in the current iteration
                Vector_Utilidades   - (list) Array with the entries of the matrix of utilities of the game (i.e the values a,b,c,d ) 
                p_i                 - (float) probability of infection of the epidemic 
                W_max  (float)      - (float) the maximum weight an edge can have
            Output:
                G                   - (networkx graph) Graph with the utilities updated after playing the game in the current iteration
            
        """
        # We assign the values of the array to the variables a,b,c,d that represent the values of the utility matrix 
        a = Vector_Utilidades[0]
        b = Vector_Utilidades[1]
        c = Vector_Utilidades[2]
        d = Vector_Utilidades[3]
        
        G = G0.copy()  # we make a copy of the graph before playing the game in the current iteration
        for nodo_i in G0.nodes():   # every node plays the game with every of its neighbors
            for nodo_j in G0.neighbors(nodo_i):             
                if G0.nodes[nodo_i]['CoD'] == 'C':            
                    if G0.nodes[nodo_j]['CoD'] == 'C':      # scenario where both nodes are Cooperators
                        G.nodes[nodo_i]['Utilidad'] += a      
                        G.nodes[nodo_j]['Utilidad'] += a
                    else:                
                        G.nodes[nodo_i]['Utilidad'] += b   # scenario where first node is Cooperator and its neighbor is Detractor
                        G.nodes[nodo_j]['Utilidad'] += c
                else:
                    if G0.nodes[nodo_j]['CoD'] == 'C':
                       G.nodes[nodo_i]['Utilidad'] += c    # scenario where first node is Detractor and its neighbor is Cooperator
                       G.nodes[nodo_j]['Utilidad'] += b
                    else:
                        G.nodes[nodo_i]['Utilidad'] += d   # scenario where both nodes are Detractors
                        G.nodes[nodo_j]['Utilidad'] += d
        # G = self.Riesgo(G,p_i,W_max)            
        return G
        
    '''
    def Exposicion_Riesgo(self,p_i,W_max,G2,nodo_i):
        e_i = 0
        infected_neighbors_i = self.intersection(self.list_of_state(G2,'I'), list(G2.neighbors(nodo_i)) )
        for nodo_j in infected_neighbors_i:
            e_i += G2[nodo_i][nodo_j]["weight"]
        e_i = e_i*p_i * 1/(W_max * G2.degree(nodo_i))
        G2.nodes[nodo_i]['Exposicion'] = float(e_i)
        return float(e_i)
    
    def Peligro_Riesgo(self,G,nodo_i):
        d_i = 0
        G0 = G.copy()
        susceptible_neighbors_i = self.intersection(self.list_of_state(G,'S'), list(G.neighbors(nodo_i)) )
        infected_neighbors_i = self.intersection(self.list_of_state(G,'I'), list(G.neighbors(nodo_i)) )
        for nodo_j in susceptible_neighbors_i:
            infected_neighbors_j = self.intersection(self.list_of_state(G,'I'), list(G.neighbors(nodo_j)) )
            d_i += len(infected_neighbors_j)
        d_i = (d_i + len(infected_neighbors_i))/G.degree(nodo_i)
        G0.nodes[nodo_i]['Peligro'] = d_i
        return d_i
    
    def Riesgo(self,G,p_i,W_max):
        G0 = G.copy()
        for nodo_i in G.nodes():
            E_i = self.Exposicion_Riesgo(p_i,W_max,G0,nodo_i)
            P_i = self.Peligro_Riesgo(G0,nodo_i)
            G0.nodes[nodo_i]['Utilidad'] = G0.nodes[nodo_i]['Utilidad'] - (P_i *E_i*G0.nodes[nodo_i]['Vulnerabilidad'] )
        return G0
    '''    
#####
#####
#####   
    
    def Epidemic_Evolution2(self,G0,p_i,r_period,p_r,inmmunity_period,p_s,incubation_period,Vector_Utilidades,W_max,h):
    
        """ Function to simulate the course of the epidemic, game and coevolution of the network over a period of time. 
    
            Input: 
                G0                  - (networkx gaph) Initial Graph
                p_i                 - (float) probability of infection of the epidemic
                r_period            - (int) number of periods of time before before an infected node can recover
                p_r                 - (float) probability of recovering from the disease
                inmmunity_period    - (int) temporary inmmunity period (minimal number of days after recovering from the disease before an indivudal can get infected again)
                p_s                 - (float) probability of becoming susceptible to the disease while being temporarily inmmune
                incubation_period   - (int) minimal number of days a node has to be infected before it can infect other nodes
                Vector_Utilidades   - (list) array with the entries of the matrix of utilities of the game 
                W_max               - (float) the maximum weight an edge can have
                h                   - (float) threshold for the change of edge weight 
                
            Output:
                Gtrack              - (list) array in which every entry is the updated graph en each step of the epidemic
                Accumulative_count  - (list) array of accumulative number of infected nodes over time
        """
        # Before the contagion of the disease begins:
        # Every node plays the Prisioner´s Dilemma with its neighbors and obtains a certain utility on the first iteration
        G0 = self.Dilema_del_Prisionero(G0,Vector_Utilidades,p_i,W_max)
        # We obtain the initial risk of every node and substract it from is utility
        G0 = self.Riesgo2(G0,p_i,W_max)
    
        Gtrack = [G0]   # initialize the array of graphs with the initial graph(introduced in the input)
        acc_inf = len(self.list_of_state(G0,'I'))   # initialize the number of accumulated infections with the number of infected in the initial graph
        Accumulative_count = [acc_inf]      # initialize the general array of accumulated infectections over time
        for step_i in tqdm(range(1,self.steps)):
        # Simulate the epidemic and update the epidemic state of the nodes in the current iteration
            Gtrack, acc_inf = self.Update_Epidemic(Gtrack,p_i,acc_inf,r_period,p_r,inmmunity_period,p_s,incubation_period)   #Update the Graphs and number of accumulated infections in every period of time
        # Every node plays the Prisioner´s Dilemma with its neighbors and obtains a certain utility on the current iteration
            Gtrack[-1] = self.Dilema_del_Prisionero(Gtrack[-1],Vector_Utilidades,p_i,W_max)
        # The Exposure of each node is calculatedfor the current iteration
            Gtrack[-1] = self.Exposicion_Riesgo2(p_i,W_max,Gtrack[-1])
        # The Danger each node perceives is calculated for the current iteration
            Gtrack[-1] = self.Peligro_Riesgo2(Gtrack[-1])
        # We calculate the Risk of each node and subtract it from the utility of the node (as the risk is considered negative utility)
            Gtrack[-1] = self.Riesgo2(Gtrack[-1],p_i,W_max)
        # The weight of the edges are changed with a process based on the nodes´ utility and behaviour
            Gtrack[-1] = self.Cambio_Peso_Enlace(Gtrack[-1],W_max,h)
        # Add the number of accumulated infections of every period to the general array
            Accumulative_count.append(acc_inf)       
    
        return Gtrack, Accumulative_count
    

    def Exposicion_Riesgo2(self,p_i,W_max,G2):
        """ Function to update the 'Exposicion' atribute for each node by using the Exposure Formula. 
    
            Input: 
                p_i     - (float) probability of infection of the epidemic
                W_max   - (float) the maximum weight an edge can have
                G2      - (networkx gaph) Graph in the current time in the simulation
                
            Output:
                G2      - (networkx gaph) Graph with 'Exposicion' atribute updated
        """
        for nodo_i in G2.nodes():   # loop through the nodes of the graph
        # initialize 'e_i' variable for a sum
            e_i = 0    
        # we obtain the list of infected neighbors of the node in the current iteration           
            infected_neighbors_i = self.intersection(self.list_of_state(G2,'I'), list(G2.neighbors(nodo_i)) )
            for nodo_j in infected_neighbors_i:   # loop through the infected neighbors
                e_i = e_i + float(G2[nodo_i][nodo_j]["weight"])    # sum the edge weights of the current iterated node with its infected neighbors
            e_i = e_i*p_i * 1/(G2.degree(nodo_i)*W_max)     # apply the exposure formula 
            G2.nodes[nodo_i]['Exposicion'] = 100*e_i        # update the 'Exposicion' atribute for the current iterated node
        return G2
    
    
    def Peligro_Riesgo2(self,G0):
        """ Function to update the 'Peligro' atribute for each node by using the Danger Formula. 
    
            Input: 
                G0   - (networkx gaph) Graph in the current time in the simulation
                
            Output:
                G0   - (networkx gaph) Graph with 'Peligro' atribute updated
        """
        for nodo_i in G0.nodes():   # loop through the nodes of the graph
        # initialize 'd_i' variable for a sum    
            d_i = 0
        # we obtain the list of susceptible neighbors of the node in the current iteration
            susceptible_neighbors_i = self.intersection(self.list_of_state(G0,'S'), list(G0.neighbors(nodo_i)) )
        # we obtain the list of infected neighbors of the node in the current iteration
            infected_neighbors_i = self.intersection(self.list_of_state(G0,'I'), list(G0.neighbors(nodo_i)) )
            for nodo_j in susceptible_neighbors_i:
            # we obtain the list of infected neighbors of the node_j inside the second loop
                infected_neighbors_j = self.intersection(self.list_of_state(G0,'I'), list(G0.neighbors(nodo_j)) )
            # sum the amount of infected neighbors of the neighbor of the node_i from the first loop
                d_i += len(infected_neighbors_j)
        
            d_i = (d_i + len(infected_neighbors_i))/G0.degree(nodo_i) # apply the danger formula
            G0.nodes[nodo_i]['Peligro'] = d_i   # update the 'Peligro atribute for the node_i from the first loop
        return G0
    
    def Riesgo2(self,G,p_i,W_max):
        """ Function to calculate the risk for every node and update its utility by substracting the risk value. 
    
            Input: 
                G   - (networkx gaph) Graph in the current time in the simulation
                
            Output:
                G0   - (networkx gaph) Graph with 'Utilidad' atribute updated
        """
        G0 = G.copy()
        for nodo_i in G0.nodes():   # loop through all the nodes in the graph
        # calculate risk value by mulitpliying danger*exposure*vulnerability 
        # then substract risk value from utility to update it
            G0.nodes[nodo_i]['Utilidad'] = G0.nodes[nodo_i]['Utilidad'] - (G0.nodes[nodo_i]['Peligro'] *G0.nodes[nodo_i]['Exposicion']*G0.nodes[nodo_i]['Vulnerabilidad'] )
        return G0
    
    
    def Choose_I_neighbor(self,W_max,G,nodo_i):
        """ Function for a susceptible node to choose an infected neighbor to change the edge weight between them.
            The selection of the infected neighbor is not totally random as the probability of selecting an 
            infected node is determined by the weight of the edge it has with the susceptible node.
            In particular this probability is defined as 
            p_ij = (weight of edge(i,j)) /sum(weights of edges between the susceptible node and its infected neighbors)
    
            Input: 
                W_max            - (float) the maximum weight an edge can have
                G                - (networkx gaph) Graph in the current time in the simulation
                nodo_i           - (networkx node) susceptible node that will choose an infected neighbor to change the edge weight between them
                
            Output:
                vecino_elegido   - (networkx node) infected neighbor chosen to change the edge weight with
        """
        W = np.array([])   # initialize empty array where edge weights will be stored
        # we obtain the list of infected neighbors of the susceptible node 'nodo_i'
        vecinos_infectados_i = self.intersection(self.list_of_state(G,'I'), list(G.neighbors(nodo_i)) )
        for nodo_j in vecinos_infectados_i:   # loop through the infected neighbors of nodo_i
            W = np.append(W,G[nodo_i][nodo_j]['weight'])   # store the edge weights between nodo_i and its infected neighbors
            # inf_n = np.array(vecinos_infectados_i)
        vecino_elegido = np.random.choice(vecinos_infectados_i, p=W/sum(W))   # choose an infected neighbor with probability p_ij
        return vecino_elegido
    
    
    def Decidir_Cambiar_Peso(self,G,nodo_i,nodo_j,h):
        """ Function that, given a susceptible node(node i) and infected node(node j) with an edge between them, 
            decides whether the weight of edge (i,j) will increase or decrease depending on both utility 
            and behavior('CoD') atributes of nodes i and j. 
    
            Input: 
                G        - (networkx gaph) Graph in the current time in the simulation
                nodo_i   - (networkx node) susceptible node that will change edge weight with nodo_j
                nodo_j   - (networkx node) infected node that will change edge weight with nodo_i
                h        - (float) threshold for the change of edge weight       
                
            Output:
                1        - indicates that the edge weight will increase
                0        - indicates that the edge weight will decrease
        """
        U_i = G.nodes[nodo_i]['Utilidad']          # asign utility value of nodo_i to variable U_i
        U_j = G.nodes[nodo_j]['Utilidad']          # asign utility value of nodo_j to variable U_j
        w_ij = float(G[nodo_i][nodo_j]['weight'])  # asign weight value of edge (i,j) to varable w_ij
        UXI = []       # initialize empty array to obtain average utility value of nodo_i´s neighbors
        
        for i in G.neighbors(nodo_i):   # loop through neighbors of nodo_i
            UXI.append( G.nodes[i]['Utilidad'])   # store utility values of nodo_i´s neighbors
        Uxi = statistics.mean(UXI) # we obtain average utility value of nodo_i´s neighbors and asign it to variable 'Uxi'
        
# the condition, based on utility, to increase or decrease the edge weight between nodes i and j
# depends on the behaviour 'CoD' atribute of the node i 
        if G.nodes[nodo_i]['CoD'] == 'C':
            if (U_i-U_j> (U_i/w_ij)) and (U_j-Uxi<h):     # condition to increase weight of edge (i,j) when node i is a Cooperator
                return 1
            else:
                return 0
        else:
            if (U_i-U_j < (U_i/w_ij)) and (U_j-Uxi>=h):   # condition to increase weight of edge (i,j) when node i is a Detractor
                return 1
            else:
                return 0
        return
    
    def Cambio_Peso_Enlace(self,G,W_max,h):
        """ Function that, for every susceptible node in the graph, chooses a link with an infected node and increases or decreases 
           the edge weight between both nodes. 
    
            Input: 
                G       - (networkx graph) Graph in the current time in the simulation
                W_max   - (float) the maximum weight an edge can have
                h       - (float) threshold for the change of edge weight
                
            Output:
                G       - (networkx graph) Graph with edge weights updated (increased or decreased in value)
        """
        # the list of susceptible nodes in the current time is obtained
        Nodos_Susceptibles = self.list_of_state(G,'S')
        for nodo_i in Nodos_Susceptibles:   # loop through the list of susceptible nodes
        # we obtain the list of infected neighbors of the susceptible node 'nodo_i'
            vecinos_infectados_i = self.intersection(self.list_of_state(G,'I'), list(G.neighbors(nodo_i)) )
        # if the current iterated susceptible node does not have any infected neighbor, then no edge weight is modified 
            if len(vecinos_infectados_i) == 0: continue   
        # an infected neighbor is chosen to change the edge weight with
            nodo_j = self.Choose_I_neighbor(W_max,G,nodo_i)
        # it is decided whether the edge weight will increase or decrease
            decision_cambio = self.Decidir_Cambiar_Peso(G,nodo_i,nodo_j,h)
            
            if decision_cambio == 1: 
                # formula for increasing the edge weight
                G[nodo_i][nodo_j]['weight'] = min( G[nodo_i][nodo_j]['weight']*max(G.nodes[nodo_i]['Utilidad'],G.nodes[nodo_j]['Utilidad'])/min(G.nodes[nodo_i]['Utilidad'],G.nodes[nodo_j]['Utilidad']), W_max )         
            else:
                # formula for decreasing the edge weight
                G[nodo_i][nodo_j]['weight'] = max( G[nodo_i][nodo_j]['weight']*min(G.nodes[nodo_i]['Utilidad'],G.nodes[nodo_j]['Utilidad'])/max(G.nodes[nodo_i]['Utilidad'],G.nodes[nodo_j]['Utilidad']), 1 )
        return G
    

########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
###################################################################################################################################################################





class Plot_Data_Epidemic(Modelo_Epidemia):
    
    def __init__(self,G_path):
        self.G_path = G_path
        return
    
    def Accumulative_Infected(self,Acc_I):
        '''
        Function to graph the Accumulated Infections in every period of time of the epidemic
        Input:
            Acc_I   - (list) array in which the ith entry represents the accumulated number of infections in the ith iteration of the simulation

        Output:
            prints a graph with the x-axis and y-axis representing time and accumulated infections over time respectively
        '''
        plt.figure()
        plt.title("Acumulated Infected Nodes Over Time")
        plt.plot(Acc_I)
        plt.xlabel("Tiempo")
        plt.ylabel("Accumulated Infections")
        return
    
    def Active_Infected(self):
        '''
        Function to graph the Accumulated Infections in every period of time of the epidemic
        Input:
            self   - G path (list) with the graph of the epidemic in the ith iteration as ith entry of the array

        Output:
            prints a graph with the x-axis and y-axis representing time and current infections over time respectively
        '''
        active_infected = []  # initialize the array with the current infected in every period of time
        for g in self.G_path:
        # the number of current infected in each period will be the same as the lenght of the list of the infected
            active_infected.append( len(self.list_of_state(g,'I')) )
            
        plt.figure()
        plt.plot(active_infected,'r')
        plt.title("Active Infected Nodes Over Time")
        plt.xlabel("Tiempo")
        plt.ylabel("Active Infected")
        return
    
    
    def paint_nodes(self,G):
        '''
        Input:
            G   - (networkx graph)
        Output:
            This function paints each node of a certain color depending on its state
            Susceptible nodes are painted green
            Infected nodes are painted red
            Recovered nodes are painted blue
    
        '''
        color_map = []
        for node in G:
            if G.nodes[node]['state'] == 'S':    # Susceptible nodes are painted in green
                color_map.append('green')
            elif G.nodes[node]['state'] == 'I':  # Infected nodes are painted in red
                color_map.append('red') 
            elif G.nodes[node]['state'] == 'R':  # Inmune nodes are painted in blue
                color_map.append('blue')
            elif G.nodes[node]['state'] == 'D':  # dead nodes are painted in black
                color_map.append('black')
        nx.draw_circular(G, node_color=color_map, with_labels=True)   # print the graph
        plt.show()
        return
    
    def grafica_pesos(self,G,LABEL):
        """ Function that makes a graph of the weight of each edge in the graph 
    
            Input: 
                G                   - (networkx gaph) Initial Graph
                LABEL               - (string) indicates the name of the label that will be printed in the histogram
               
            Output:
                prints graph with the weight of each edge
        """
        Pesos = []   # initialize empty array where edge weights will be stored
        for u,v,w in G.edges(data=True):     # loop through edges in the graph
            Pesos.append(float(w['weight']))   # the edge weight currently iterated is stored in the array
            
        plt.plot(Pesos,label=LABEL)      # create graph
        plt.xlabel('numero de enlace')   # label for the x-axis
        plt.ylabel('peso de enlace')     # label for the y-axis
        plt.title('Grafica de pesos de los enlaces')    # title of the graph
        plt.legend()
        plt.show()
        return   

    def histograma_pesos(self,G,LABEL,W_max):
        """ Function to create a histogram for the edge weight distribution 
    
            Input: 
                G                   - (networkx gaph) Initial Graph
                LABEL               - (string) indicates the name of the label that will be printed in the histogram
                W_max               - (float) the maximum weight an edge can have
                
            Output:
                prints histogram with the edge weight distribution
        """
        Pesos = []   # initialize empty array where edge weights will be stored
        for u,v,w in G.edges(data=True):     # loop through edges in the graph
            Pesos.append(float(w['weight']))   # the edge weight currently iterated is stored in the array
            
        num_bins = int(W_max)   # the number of bins of the histogram is defined by the maximum weight an edge can have 
        n, bins, patches = plt.hist(Pesos, num_bins, alpha=0.5, label=LABEL)  # the histogram is created
        plt.show()          
        plt.xlabel('peso del enlace')       # label for the x-axis
        plt.ylabel('cantidad de enlaces')   # label for the y-axis
        plt.title('Distribucion de pesos de los enlaces')   # title of the graph
        plt.legend()
        plt.show()
        return 

########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################


class Save_Info(object):
    """ docstring for Save_Info"""

    def __init__(self):     #inicializamos self
        return

    def MakeDir(self,path_name):       #funcion para crear un directorio (carpeta)

    	if not os.path.exists(path_name):  #si la carpeta con el nombre 'path_name' no existe
    		os.makedirs(path_name)         # se crea una carpeta con el nombre deseado 'path_name'
    	else:
    		shutil.rmtree(path_name)       # si existe, entonces se usa ese directorio ya creado
    		os.makedirs(path_name)

    	return

    def SaveFiles(self,path_name,make_dir=False,file_format='*.hdf5'):
    
        """
          Routine to Copy all the files with a given format into the folder path_name

          path_name: The destination folder
          format:    the files format that will be copy to path_name

        """
    
        if make_dir == True:
            self.MakeDir(path_name)
        else: pass

        #formato1 = '*.hdf5'
        #files1 = glob.iglob(os.path.join(file_format))
        #for ifile in files1: shutil.move(ifile,path_name)

        # Copy  the files in format file_format to path_name
        for ifile in glob.iglob(os.path.join(file_format)):
            # Si el archivo for formato 'file_format' ya existe, lo elimina y guarda el nuevo
            if os.path.isfile(ifile):
                try:
                    shutil.move(ifile,path_name)
                except:
                    os.remove(path_name+'/'+ifile)
                    shutil.move(ifile,path_name)

        return


########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

'''
class Juegos(object):

    def __init__(self):     #inicializamos self
        return
    

    def Exposicion_Riesgo2(self,p_i,W_max,G2):
        for nodo_i in G2.nodes():
            aux = 0
            infected_neighbors_i = Modelo_Epidemia.intersection(Modelo_Epidemia.list_of_state(G2,'I'), list(G2.neighbors(nodo_i)) )
            for nodo_j in infected_neighbors_i:
                aux = aux + G2[nodo_i][nodo_j]["weight"]
            e_i = aux*p_i * 1/(G2.degree(nodo_i)*W_max)#(W_max *G2.degree(nodo_i))
            G2.nodes[nodo_i]['Exposicion'] = float(100*e_i)
        return G2
    
    
    def Peligro_Riesgo2(self,G0):
        for nodo_i in G0.nodes():
            d_i = 0
            susceptible_neighbors_i = Modelo_Epidemia.intersection(Modelo_Epidemia.list_of_state(G0,'S'), list(G0.neighbors(nodo_i)) )
            infected_neighbors_i = Modelo_Epidemia.intersection(Modelo_Epidemia.list_of_state(G0,'I'), list(G0.neighbors(nodo_i)) )
            for nodo_j in susceptible_neighbors_i:
                infected_neighbors_j = Modelo_Epidemia.intersection(Modelo_Epidemia.list_of_state(G0,'I'), list(G0.neighbors(nodo_j)) )
                d_i += len(infected_neighbors_j)
            d_i = (d_i + len(infected_neighbors_i))/G0.degree(nodo_i)
            G0.nodes[nodo_i]['Peligro'] = d_i
        return G0
    
    def Riesgo2(self,G,p_i,W_max):
        G0 = G.copy()
        for nodo_i in G0.nodes():
            G0.nodes[nodo_i]['Utilidad'] = G0.nodes[nodo_i]['Utilidad'] - (G0.nodes[nodo_i]['Peligro'] *G0.nodes[nodo_i]['Exposicion']*G0.nodes[nodo_i]['Vulnerabilidad'] )
        return G0
    
   
    def Dilema_del_Prisionero(self,G0,Vector_Utilidades,p_i,W_max):
        """ This function simulates the game (based no the Prisioner´s Dilemma) that nodes play with their neighbors
            to gain 'utility' in every iteration. Their profit will depend on their election of strategy. For this 
            particular game there are only 2 choices: 'C' or 'D' which stand for Cooperator or Detractor respectively. 
    
            Input: 
                G0                   - Graph before playing in the current iteration
                Vector_Utilidades    - Array with the entries of the matrix of utilities of the game (i.e the values a,b,c,d ) 
            Output:
                G                    - Graph with the utilities updated after playing the game in the current iteration
            
        """
        # We assign the values of the array to the variables a,b,c,d that represent the values of the utility matrix 
        a = Vector_Utilidades[0]
        b = Vector_Utilidades[1]
        c = Vector_Utilidades[2]
        d = Vector_Utilidades[3]
        
        G = G0.copy()  # we make a copy of the graph before playing the game in the current iteration
        for nodo_i in G.nodes():   # every node plays the game with every of its neighbors
            for nodo_j in G.neighbors(nodo_i):             
                if G.nodes[nodo_i]['CoD'] == 'C':            
                    if G.nodes[nodo_j]['CoD'] == 'C':      # scenario where both nodes are Cooperators
                        G.nodes[nodo_i]['Utilidad'] += a      
                        G.nodes[nodo_j]['Utilidad'] += a
                    else:                
                        G.nodes[nodo_i]['Utilidad'] += b   # scenario where first node is Cooperator and its neighbor is Detractor
                        G.nodes[nodo_j]['Utilidad'] += c
                else:
                    if G.nodes[nodo_j]['CoD'] == 'C':
                       G.nodes[nodo_i]['Utilidad'] += c    # scenario where first node is Detractor and its neighbor is Cooperator
                       G.nodes[nodo_j]['Utilidad'] += b
                    else:
                        G.nodes[nodo_i]['Utilidad'] += d   # scenario where both nodes are Detractors
                        G.nodes[nodo_j]['Utilidad'] += d
        G = self.Riesgo(G,p_i,W_max)            
        return G
    
    
    def Exposicion_Riesgo(self,p_i,W_max,G,nodo_i):
        e_i = 0
        infected_neighbors_i = Modelo_Epidemia.intersection(Modelo_Epidemia.list_of_state(G,'I'), list(G.neighbors(nodo_i)) )
        for nodo_j in infected_neighbors_i:
            e_i += G[nodo_i][nodo_j]["weight"]
        e_i = e_i*p_i * 1/(W_max * G.degree(nodo_i))
        G.nodes[nodo_i]['Exposicion'] = e_i
        return G
    
    def Peligro_Riesgo(self,G,nodo_i):
        d_i = 0
        susceptible_neighbors_i = Modelo_Epidemia.intersection(Modelo_Epidemia.list_of_state(G,'S'), list(G.neighbors(nodo_i)) )
        infected_neighbors_i = Modelo_Epidemia.intersection(Modelo_Epidemia.list_of_state(G,'I'), list(G.neighbors(nodo_i)) )
        for nodo_j in susceptible_neighbors_i:
            infected_neighbors_j = Modelo_Epidemia.intersection(Modelo_Epidemia.list_of_state(G,'I'), list(G.neighbors(nodo_j)) )
            d_i += len(infected_neighbors_j)
        d_i = (d_i + len(infected_neighbors_i))/G.degree(nodo_i)
        G.nodes[nodo_i]['Peligro'] = d_i
        return G
    
    def Riesgo(self,G,p_i,W_max):
        for nodo_i in G.nodes():
            self.Exposicion_Riesgo(p_i,W_max,G,nodo_i)
            self.Peligro_Riesgo(G,nodo_i)
            G.nodes[nodo_i]['Utilidad'] = G.nodes[nodo_i]['Utilidad'] - (G.nodes[nodo_i]['Peligro'] *G.nodes[nodo_i]['Exposicion']*G.nodes[nodo_i]['Vulnerabilidad'] )
        return G
'''