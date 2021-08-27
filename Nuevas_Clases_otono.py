import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

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
                        G               - networkx grpah with 'state' atribute (refering to the state of a node)
                        state_of_nodes  - Character (S,I,R or D) refereing to the state of a set of nodes of interest 
            Output:
                        list of nodes(natural numbers) with the certain state of interest (S,I,R,D)
        '''
        
        nodes_dict =  dict(G.nodes(data='state'))  # we obtain the dictionary with keys as nodes 
        # then we run a loop over the keys of the dictionary to select the nodes with the respective state introduced in the functions input
        return [nodo  for (nodo,value) in nodes_dict.items() if value == state_of_nodes] 
    
    
    
    def change_state(self,Nodes_of_Certain_State,G,new_state):
        '''
        This Function changes the state of the nodes of a certain group of nodes introduced in the input 
            Input:
                    Nodes_of_Certain_State - List of nodes that will change state
                    G                      - Graph in which the nodes will change their state
                    new_state              - the state that the nodes will change into
            Output:
                    G                      - Graph with the nodes states updated 
        '''
        
        for nodo in Nodes_of_Certain_State:  # we make a loop through the list of the nodes that want to be changed
            G.nodes[nodo]['state'] = new_state  # the states are changed to the input variable "new_state"
        
        return G
    
    
    def intersection(self,lst1, lst2):
        '''
        This function gives the intersection between two lists
        
        Input: 
                 lst1 - fisrt list
                 lst2 - second list
        Output:
                 lst3 - intersection between fisrt and second lists
        '''
        
        lst3 = []           # we initialize the list where both lists will intersect
        for value in lst1:      # loop through the first list
            if value in lst2:   # if an element of the first list is also in the second list
                lst3.append(value)  # that element is added to the list of the intersection
        return lst3   # it returns the list of all elements that belong to both lists
    

########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
#########################################################################################################################################################
    

    def Update_State2(self,G_array,p_i,acc_inf,recoverable_period,p_r):
        '''
        This Function updates the state of the nodes of the graph with a stochastical process
        It selects the nodes with an infected state ('state=I') and then it makes a list of the susceptible neighbors
        of that infected node. With a binomial distribution, it then selects how many nodes will be infected by an
        infected individual. Finally, it selects a random sample from the susceptible neighbors 
        with the size obtained by the binoamial distribution before mentioned.
        It also adds the accumulative amount of infected nodes in every period of time
        Then it makes a list of infected nodes that are able (after a certain amount of periods of being infected)
        to recover. After making that list, it selects a random sample of those infected nodes to change their
        status from infected to revered.
    
        Input:
                    G_array             - array of graphs of the epidemic over time 
                    p_i                 - probability of infection 
                    acc_inf             - accumulated number of infected nodes
                    recoverable_period  - number of periods of time before before an infected node can recover
                    iteration           - number of iteration(period) of the main epidemic process
                    p_r                 - probability of recovering when infected
        Output:
                    G_array             - array of graphs representing the epidemic over time 
                    acc_inf             - updated number of accumulated infected nodes 
        '''
        G = G_array[-1]     # select the last graph of the array 
                            # this represents the graph of the current period of time
        G_next = G.copy()           # create a copy of the current graph
        iteration = len(G_array)    # the iteration or current period of time will be the same as the lenght of the array so far
    
        
        for infected_node in self.list_of_state(G,'I'):  # loop through the list of current infected nodes
        # we obtain the list of suceptible neighbors of each infected node by intersecting the list 
        # of susceptible nodes and the list of the neighbors of an infected node
            susceptible_neighbors = self.intersection(self.list_of_state(G_next,'S'), list(G_next.neighbors(infected_node)) )
            
            if len(susceptible_neighbors) == 0:     # if the infected node has no susceptible neighbors
                continue                            # we continue through the loop to the next infected node
            else:
                # if the infected node has at least one suscpetible neighbor 
                # first we obtain the number of neighbors that will get infected by an infected node with a binomial process
                
            # binomial with parameters n=number of susceptible neighbors and p=probability of infection (introduced in the input)
                num_new_inf_neighbors = np.random.binomial(len(susceptible_neighbors), p_i)   
                
                if num_new_inf_neighbors == 0:   # if a node doesn´t infect anyone we continue through the loop of infected nodes
                    continue      
                else:
                # we select a random sample from the susceptible neighbors of the size of "num_new_inf_neighbors" 
                    new_inf_neighbors = random.sample(susceptible_neighbors, num_new_inf_neighbors)
                # the number of accumulated infected is updated by adding the number of new infected by a node
                    acc_inf = acc_inf + num_new_inf_neighbors
                # we change the state of the new infected from 'S' susceptible to 'I' infected
                    self.change_state(new_inf_neighbors,G_next,'I')
                    
                    
        ###################################  RECOVERING SECTION #############################################
        ###############################################################################################
        if iteration >= recoverable_period:         # if the period of time is smaller than the recoverable period, no nodes can recover yet
            Possible_Recovered = self.list_of_state(G_next,'I')  # we initialize "Possible_Recovered" as the list of current infected 
            
        # loop to obtain the list of nodes that are currently infected and were infected up until
        # a "recoverable_period" in the past
            for i in range(0,recoverable_period):    
            #Update the Possible Recovered by intersecting that list with the list of nodes that were infected i periods in the past
                Possible_Recovered = self.intersection(self.n_steps_before_state(G_array,'I',i), Possible_Recovered)
            
            if len(Possible_Recovered)!=0:  #if there are nodes that can recover from infection
            # with a binomial process we obtain the number of infected nodes that will recover 
                
            # binomial with parameters n=number of nodes that can recover and p=probability of recovering (introduced in the input)
                num_recovered = np.random.binomial(len(Possible_Recovered), p_r)
            # we select a random sample from the Possible Recovered nodes with the size of "num_recovered"     
                Recovered = random.sample(Possible_Recovered,num_recovered)
            # we change the state of the nodes that recover from 'I' infected to 'R' recovered
                self.change_state(Recovered,G_next,'R')
        # the Graph is updated by changing the state of the new infected and the recovered nodes
        G_array.append(G_next)
        
        return G_array,acc_inf
    
    
    
    
    def Epi_Evolution(self,G0,p_i,r_period,p_r):
    
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
    
        Gtrack = [G0]   # initialize the array of graphs with the initial graph(introduced in the input)
        acc_inf = len(self.list_of_state(G0,'I'))   # initialize the number of accumulated infections with the number of infected in the initial graph
        Accumulative_count = [acc_inf]      # initialize the general array of accumulated infectections over time
        for step_i in tqdm(range(1,self.steps)):
            Gtrack, acc_inf = self.Update_State2(Gtrack,p_i,acc_inf,r_period,p_r)   #Update the Graphs and number of accumulated infections in every period of time
            Accumulative_count.append(acc_inf)      # Add the number of accumulated infections of every period to the general array 
    
        return Gtrack, Accumulative_count
    
    
    
    
    def Initial_Graph(self,N_infected,graph_name):
        '''
        Function to define the initial graph in which it randomly selects which nodes will start as infected or 
        suscpetible with the next graph atribute
        
        'state' = String 'S' (susceptible) or 'I' (infected)
    
        Input:
                    N              - Number of nodes
                    N_infected     - Number of initial infected nodes
    
        Output:
                    G              - the graph with the atribute 'state'
    
        '''
        
        if graph_name == 'ER'   : G = nx.erdos_renyi_graph(self.N,0.3)
        if graph_name == 'NWS'  : G = nx.newman_watts_strogatz_graph(self.N,4,0.3)
        if graph_name == 'BA'   : G = nx.barabasi_albert_graph(self.N,2)
    
    
    
        # Define una etiqueta para cada nodo
        for nodo in G.nodes():
            #G.nodes[nodo]['id']     = nodes_labels[]                                           # Define la etiqueta de cada nodo
            #G.nodes[(i,j)]['Nd']     = random.choice( range(Num_Diaforinas[0],Num_Diaforinas[1]) )   # Define el número de vectores 
            G.nodes[nodo]['state'] = 'S'                                                           # Define el estado 'S'
    
        # Rutina para seleccionar N_infected nodes del grafo y cambiar su estado a 'I'
        seeds  = random.sample(list(G.nodes()), N_infected)
    
        # Ciclo sobre los árboles seleccionados aleatoriamente
        for nodo in seeds: G.nodes[nodo]['state'] = 'I'
    
        return G
    
    
    
    def n_steps_before_state(self,Graph_Array,state_of_nodes,n_steps):
        '''
        Function to search for the list of nodes that had a certain state n steps in the past
        
        Posible states: 'S' (susceptible), 'I' (infected), 'R' (recovered), 'D' (deceased)
    
        Input:
                    Graph_Array       - Array that contains each graph updated every period of time in the epidemic
                    state_of_nodes    - String of State of the nodes that are being searched for 
                    n_steps           - Number of steps into the past where the searching will occur
    
        Output:
                    searched_nodes    - the list of nodes that had a certain state n steps in the past
    
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
    
    def Initial_Graph_atribute(self,N_infected,graph_name):
        '''
        Function to define the initial graph in which it randomly selects which nodes will start as infected or 
        suscpetible with the next graph atribute
        
        'state' = String 'S' (susceptible) or 'I' (infected)
    
        Input:
                    N              - Number of nodes (global variable)
                    N_infected     - Number of initial infected nodes
    
        Output:
                    G              - the graph with the atribute 'state'
    
        '''
        # section to select the type of graph that will be used for the simulation
        if graph_name == 'ER'   : G = nx.erdos_renyi_graph(self.N,0.3)
        if graph_name == 'NWS'  : G = nx.newman_watts_strogatz_graph(self.N,4,0.3)
        if graph_name == 'BA'   : G = nx.barabasi_albert_graph(self.N,2)
    
        # Once the graph is created, we define the atributes for each node
        # There are three atributes: 'state'  - which defines the state of the node in the epidemic (Susceptible,Infected,Recovered,Deceased)
        #                            'days_I' - amount of days that a node has been infected
        #                            'days_R' - amount of days that a node has been recovered
        for nodo in G.nodes():
            G.nodes[nodo]['state']  = 'S'       
            G.nodes[nodo]['days_I'] = 0
            G.nodes[nodo]['days_R'] = 0
    
        # A random sample of size 'N_infected' is selected from the node
        seeds  = random.sample(list(G.nodes()), N_infected)
    
        # Loop through the intial infected to change their state from 'S' to 'I'
        for nodo in seeds: G.nodes[nodo]['state'] = 'I'
        # the number of days infected for the first graph are initialized
        self.add_or_reset_state_days(seeds, G, 'days_I','add')
    
        return G
    
    def add_or_reset_state_days(self,Nodes_of_Certain_State,G,I_R_days,add_or_reset):
        '''
        This Function adds(+1) or resets(=0) the number of days that a node has been in a certain state
            Input:
                    Nodes_of_Certain_State - List of nodes that will change state
                    G                      - Graph in which the nodes will change their state
                    I_R_days               - (string) state of the number of days that will be updated
                    add_or_reset           - (string) 'add' will add a day to the number of days and 'reset' will make it 0
            Output:
                    G                      - Graph with the nodes days with a certain state updated 
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
                    G               - Current graph of the epidemic
                    r_period        - minimal number of days a node has to be in a certain state to be included in the output list
                    state_counted   - (string= 'I' or 'R') the state in which the list of nodes have been for a certain time
        '''
        List_Nodes_Days_I_R = []   # we initialize the list of nodes 
        atribute = 'days_'+state_counted     # create a string by adding 'days' to the state introduced in the input
        for person in self.list_of_state(G, state_counted):   # Loop through the list of Infected or Recovered in the graph
            if G.nodes[person][atribute] >= r_period:  # if the number of days of the state of interest in a node is at least r_period days
                List_Nodes_Days_I_R.append(person)     # the node is added to the list of nodes of the output
        return List_Nodes_Days_I_R
    
        
    def Update_Epidemic(self,G_array,p_i,acc_inf,recoverable_period,p_r):
        '''
        This Function updates the state of the nodes of the graph with a stochastical process
        It selects the nodes with an infected state ('state=I') and then it makes a list of the susceptible neighbors
        of that infected node. With a binomial distribution, it then selects how many nodes will be infected by an
        infected individual. Finally, it selects a random sample from the susceptible neighbors 
        with the size obtained by the binoamial distribution before mentioned.
        It also adds the accumulative amount of infected nodes in every period of time
        Then it makes a list of infected nodes that are able (after a certain amount of periods of being infected)
        to recover. After making that list, it selects a random sample of those infected nodes to change their
        status from infected to recovered.
    
        Input:
                    G_array             - array of graphs of the epidemic over time 
                    p_i                 - probability of infection 
                    acc_inf             - accumulated number of infected nodes
                    recoverable_period  - number of periods of time before before an infected node can recover
                    iteration           - number of iteration(period) of the main epidemic process
                    p_r                 - probability of recovering when infected
        Output:
                    G_array             - array of graphs representing the epidemic over time 
                    acc_inf             - updated number of accumulated infected nodes 
        '''
        G = G_array[-1]     # select the last graph of the array 
                            # this represents the graph of the current period of time
        G_next = G.copy()           # create a copy of the current graph
        #iteration = len(G_array)    # the iteration or current period of time will be the same as the lenght of the array so far
    
        
        for infected_node in self.list_of_state(G_next,'I'):  # loop through the list of current infected nodes
        # we obtain the list of suceptible neighbors of each infected node by intersecting the list 
        # of susceptible nodes and the list of the neighbors of an infected node
            susceptible_neighbors = self.intersection(self.list_of_state(G_next,'S'), list(G_next.neighbors(infected_node)) )
            
            if len(susceptible_neighbors) == 0:     # if the infected node has no susceptible neighbors
                continue                            # we continue through the loop to the next infected node
            else:
                # if the infected node has at least one suscpetible neighbor 
                # first we obtain the number of neighbors that will get infected by an infected node with a binomial process
                
            # binomial with parameters n=number of susceptible neighbors and p=probability of infection (introduced in the input)
                num_new_inf_neighbors = np.random.binomial(len(susceptible_neighbors), p_i)   
                
                if num_new_inf_neighbors == 0:   # if a node doesn´t infect anyone we continue through the loop of infected nodes
                    continue      
                else:
                # we select a random sample from the susceptible neighbors of the size of "num_new_inf_neighbors" 
                    new_inf_neighbors = random.sample(susceptible_neighbors, num_new_inf_neighbors)
                # the number of accumulated infected is updated by adding the number of new infected by a node
                    acc_inf = acc_inf + num_new_inf_neighbors
                # we change the state of the new infected from 'S' susceptible to 'I' infected
                    self.change_state(new_inf_neighbors,G_next,'I')
        
                    
        ###################################  RECOVERING SECTION #############################################
        ###############################################################################################
        
        # we obtain the list of nodes that have been infected for at least "rrecoverable_period" amount of days
        Possible_Recovered = self.Count_Days_I_R(G_next, recoverable_period, 'I')
        
        
        if len(Possible_Recovered)!=0:  #if there are nodes that can recover from infection
        # with a binomial process we obtain the number of infected nodes that will recover 
            
        # binomial with parameters n=number of nodes that can recover and p=probability of recovering (introduced in the input)
            num_recovered = np.random.binomial(len(Possible_Recovered), p_r)
        # we select a random sample from the Possible Recovered nodes with the size of "num_recovered"     
            Recovered = random.sample(Possible_Recovered,num_recovered)
        # we change the state of the nodes that recover from 'I' infected to 'R' recovered
            self.change_state(Recovered,G_next,'R')
        # the Graph is updated by changing the state of the new infected and the recovered nodes
        G_array.append(G_next)
        
        # we add a day to the atribute 'days_I' to all the nodes currently infected at the end of the period
        self.add_or_reset_state_days(self.list_of_state(G_next,'I'), G_next, 'days_I','add')
        # we add a day to the atribute 'days_R' to all the nodes currently recovered at the end of the period
        self.add_or_reset_state_days(self.list_of_state(G_next,'R'), G_next, 'days_R','add')
        
        # we reset the atribute 'days_I' to 0 of the recovered nodes
        self.add_or_reset_state_days(self.list_of_state(G_next,'R'), G_next, 'days_I','reset')
        return G_array,acc_inf
    
    
    def Epidemic_Evolution(self,G0,p_i,r_period,p_r):
    
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
    
        Gtrack = [G0]   # initialize the array of graphs with the initial graph(introduced in the input)
        acc_inf = len(self.list_of_state(G0,'I'))   # initialize the number of accumulated infections with the number of infected in the initial graph
        Accumulative_count = [acc_inf]      # initialize the general array of accumulated infectections over time
        for step_i in tqdm(range(1,self.steps)):
            Gtrack, acc_inf = self.Update_Epidemic(Gtrack,p_i,acc_inf,r_period,p_r)   #Update the Graphs and number of accumulated infections in every period of time
            Accumulative_count.append(acc_inf)      # Add the number of accumulated infections of every period to the general array 
    
        return Gtrack, Accumulative_count



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
        It requires the array of Accumulated Infections in the input

        '''
        plt.figure()
        plt.title("Acumulated Infected Nodes Over Time")
        plt.plot(Acc_I)
        plt.xlabel("Tiempo")
        plt.ylabel("Accumulated Infections")
        return
    
    def Active_Infected(self):
        '''
        Function to graph the Current Infected nodes in every period of time in the Epidemic
        it requires in the input the array of the graphs of the epidemic over time

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
        This function paints each node of a certain color depending on its state 
    
        '''
        color_map = []
        for node in G:
            if G.nodes[node]['state'] == 'S':
                color_map.append('green')
            elif G.nodes[node]['state'] == 'I': 
                color_map.append('red') 
            elif G.nodes[node]['state'] == 'R': 
                color_map.append('blue')
            elif G.nodes[node]['state'] == 'D': 
                color_map.append('black')
        nx.draw_circular(G, node_color=color_map, with_labels=True)
        plt.show()
        return

########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

