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
    
    def __init__(self,N,steps):  #inicializamos el self, con parametros globales
        self.N = N           # numero de nodos
        self.steps = steps   # numero de pasos o frames
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
        
        nodes_dict =  dict(G.nodes(data='state')) 
        return [nodo  for (nodo,value) in nodes_dict.items() if value == state_of_nodes]
    
    
    '''
    def list_susceptible(self,G):
        
        nodes_dict =  dict(G.nodes(data='state')) 
        return [nodo  for (nodo,value) in nodes_dict.items() if value == 'S']
    
    
    
    def list_recovered(G):
        
        nodes_dict =  dict(G.nodes(data='state')) 
        return [nodo  for (nodo,value) in nodes_dict.items() if value == 'R']
    
    
    
    def list_deaths(G):
        
        nodes_dict =  dict(G.nodes(data='state')) 
        return [nodo  for (nodo,value) in nodes_dict.items() if value == 'D']
    '''
    
    
    def Update_State(self,G,p):
        '''
        This Function updates the state of the nodes of the graph with a stochastical process
        It selects the nodes with an infected state ('state=I') and then it makes a list of the susceptible neighbors
        of that infected node, it runs a loop to select the neighbors that will get infected and change their state
        from 'S' to 'I'
    
        Input:
                    G - the graph in the actual period of time
                    p - probability of infection 
        Output:
                    G_next - the graph in te next period of time 
        '''
    
        G_next = G.copy()
    
        # Loop over the list of infected 
        for infected_node in self.list_of_state(G,'I'):
    
            #Loop over the neighbors of an infected node
            for neighbor_i in list(G.neighbors(infected_node)):
    
                # If the node of an infected is already infected
                if G.nodes[neighbor_i]['state'] == 'I' :
                    continue
                else:  # it finds the susceptible neighbors ans asks if it will get infected with a probability p
                    if random.random() < p:
                        G_next.nodes[neighbor_i]['state'] = 'I'
    
        return G_next
    
    
    
    def intersection(self,lst1, lst2):
        '''
        This function gives the intersection between two lists
        
        Input: 
                 lst1 - fisrt list
                 lst2 - second list
        Output:
                 lst3 - intersection between fisrt and second lists
        '''
        
        lst3 = []
        for value in lst1:
            if value in lst2:
                lst3.append(value)
        return lst3
    
    
    
    def Update_State2(self,G,p,acc_inf):
        '''
        This Function updates the state of the nodes of the graph with a stochastical process
        It selects the nodes with an infected state ('state=I') and then it makes a list of the susceptible neighbors
        of that infected node. With a binomial distribution, it then selects how many nodes will be infected by an
        infected individual. Finally, it selects a random sample from the susceptible neighbors 
        with the size obtained by the binoamial distribution before mentioned. 
    
        Input:
                    G - the graph in the actual period of time
                    p - probability of infection 
        Output:
                    G_next - the graph in te next period of time 
        '''
    
        G_next = G.copy()
    
        # Ciclo sobre el conjunto de nodos (árboles) actualmente infectados
        for infected_node in self.list_of_state(G,'I'):
            susceptible_neighbors = self.intersection(self.list_of_state(G,'S'), list(G.neighbors(infected_node)) )
            
            if len(susceptible_neighbors) == 0:
                continue
            else:
                num_vecinos_contagiados_i = np.random.binomial(len(susceptible_neighbors), p)  
                acc_inf = acc_inf + num_vecinos_contagiados_i
                if num_vecinos_contagiados_i == 0:         # si no se contagia a nadie en ese periodo
                    continue      
                else:
                    vecinos_contagiados_i = random.sample(susceptible_neighbors, num_vecinos_contagiados_i)
    
    
                for neighbor_i in vecinos_contagiados_i:
                    G_next.nodes[neighbor_i]['state'] = 'I'
        
        return G_next,acc_inf
    
    
    def Epi_Evolution(self,G0,p):
    
        """ Function to simulate the course of the epidemic over a period of time. 
    
            Input: 
                G0     - Initial Graph
                steps  - number of steps of the simulation
            Output:
                Gtrack  - array in which every entry is the updated graph en each step of the epidemic
        """
    
        Gtrack = [G0]
        acc_inf = 1
        Accumulative_count = [acc_inf]
        for step_i in tqdm(range(1,self.steps)):
            graph_to_append, acc_inf = self.Update_State2(Gtrack[step_i-1],p,acc_inf)
            Accumulative_count.append(acc_inf)
            Gtrack.append( graph_to_append ) 
    
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
    
    
    def n_steps_before_state(self,Graph_Array,state_of_nodes,n_steps):
        '''
        Function to search for the list of nodes that had a certain state n steps in the past
        
        Posible states: 'S' (susceptible), 'I' (infected), 'R' (recovered), 'D' (deceased)
    
        Input:
                    Graph_Array       - Array that contains each graph updated every period of time in the epidemic
                    state_od_nodes    - State of the nodes that are being searched for 
                    n_steps           - Number of steps into the past where the searching will occur
    
        Output:
                    searched_nodes    - the list of nodes that had a certain state n steps in the past
    
        '''
        
        total_lenght = len(Graph_Array)
        if n_steps>=total_lenght:
            return
        
        G = Graph_Array[total_lenght-1 - n_steps]
        searched_nodes = self.list_of_state(G,state_of_nodes)
        
        return searched_nodes


########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################



'''
G3 = Initial_Graph(5,2,"ER")
p = 1


####################################################################################
G1 = nx.Graph(tipo="state")
G1.add_node(0,state="S")
G1.add_node(1,state="S")
G1.add_node(2,state="I")
G1.add_node(3,state="S")
G1.add_node(4,state="I")

G1.add_edge(1,2)
G1.add_edge(2,3)
#G1.add_edge(3,4)
#G1.add_edge(4,5)
G1.add_edge(5,1)

nx.draw_circular(G1,with_labels=True)

print(list_susceptible(G1))

print(list_infected(G1))


G2 = Update_State2(G1,p)
print(list_infected(G2))
#########################################################################################


plt.figure()
nx.draw_circular(G3,with_labels=True)
print("\n")
print("\n")
print(list_susceptible(G3))

print(list_infected(G3))


G4 = Update_State2(G3,p)
print(list_infected(G4))

print("\n")
print("\n")




N          = 20    
steps      = 10
p          = 0.1

G0  = Initial_Graph(N,1,'ER')


# nodes_size = 30          # [ j for i,j in Go.nodes(data='Nd')]

G_path     = Epi_Evolution(G0,steps,p)

for t in G_path:
    plt.figure()
    paint_nodes(t)
'''



