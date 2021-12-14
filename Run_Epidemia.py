from Nuevas_Clases_otono_Juegos import Modelo_Epidemia
from Nuevas_Clases_otono_Juegos import Plot_Data_Epidemic
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np

# Global Parameters
N          = 20    
steps      = 50
W_max      = 30
# Calling of Classes
ME  = Modelo_Epidemia(N,steps)


# Parameters of the Epidemic
p_i          = 0.2
p_r          = 0.7
p_s          = 0.9
periodo_infeccion = 14
periodo_inmune       = 21
initial_infected = 1
periodo_incubacion = 10

# Parameters of the Game
Vector_Utilidades = [6,1,7,2]
detractor_prob = 0.1
h = 5


# we obtain the initial Graph of the epdimedic in the first period of time
G0  = ME.Initial_Graph_atribute(initial_infected,'ER',detractor_prob,W_max)

# the simulation of the epidemic takes place
G_path, ACC_inf     = ME.Epidemic_Evolution2(G0,p_i,periodo_infeccion,p_r,periodo_inmune,p_s,periodo_incubacion,Vector_Utilidades,W_max,h)

# the array of graphs is saved into a pkl archive
#outfile = open("Graphs_path.pkl",'wb')
#pkl.dump(G_path,outfile)
#outfile.close()
#Data = pkl.load(open('Graphs_path.pkl','rb'))
#Graphs = Data

file_name = "Epidemic_Graphs_N"+str(N)+"_t"+str(steps)+".pkl"
outfile = open(file_name,'wb')
pkl.dump(G_path,outfile)
outfile.close()
Data = pkl.load(open(file_name,'rb'))


PDE = Plot_Data_Epidemic(G_path)
# each graph is printed by painting nodes with different colors depending in their state
# for t in G_path:
    # plt.figure()
    # PDE.paint_nodes(t)
    #for nodo in t:
    #print(t.nodes[5]['days_I'])
    #print(t.nodes[5]['days_R'])
    #print(t.nodes[5]['state'])
    #print('\n')    

# we create graphs to visualize the accumulated infections and the current infections over time
PDE.Accumulative_Infected(ACC_inf)
PDE.Active_Infected()

#for nodo in Graphs[-1].nodes():
#    print(Graphs[-1].nodes[nodo]['days_I'])
#    print(Graphs[-1].nodes[nodo]['days_R'])
#    print('\n')
#print(ME.list_of_state(G_path[-1],'I'))

GG = G_path[-1]

#GG = ME.Riesgo(GG,p_i,W_max)
# for i in GG.nodes():
    # print(GG.nodes[i]['CoD'])
    # print(GG.nodes[i]["Utilidad"])
    # print(GG.nodes[i]['Peligro'] )
    # print(GG.nodes[i]['Exposicion'] )

plt.figure()    
PDE.grafica_pesos(G_path[0],'Pesos Iniciales')
PDE.grafica_pesos(GG,'Pesos Finales')

plt.figure()    
PDE.histograma_pesos(G_path[0],'Pesos Iniciales',W_max)
PDE.histograma_pesos(GG,'Pesos Finales',W_max)

plt.figure()
PDE.paint_nodes(G_path[0])


