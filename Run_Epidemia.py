from Nuevas_Clases_otono import Modelo_Epidemia
from Nuevas_Clases_otono import Plot_Data_Epidemic
import matplotlib.pyplot as plt
import pickle as pkl

# Global Parameters
N          = 10    
steps      = 10
# Calling of Classes
ME  = Modelo_Epidemia(N,steps)


# Parameters of the Epidemic
p_i          = 1
p_r          = 1
periodo_recuperacion = 3
initial_infected = 1

# we obtain the initial Graph of the epdimedic in the first period of time
G0  = ME.Initial_Graph3(initial_infected,'ER')

# the simulation of the epidemic takes place
G_path, ACC_inf     = ME.Epi_Evolution3(G0,p_i,periodo_recuperacion,p_r)

# the array of graphs is saved into a pkl archive
outfile = open("Graphs_path.pkl",'wb')
pkl.dump(G_path,outfile)
outfile.close()
Data = pkl.load(open('Graphs_path.pkl','rb'))
Graphs = Data

# each graph is printed by painting nodes with different colors depending in their state
#for t in Graphs:
#    plt.figure()
#    ME.paint_nodes(t)

# we create graphs to visualize the accumulated infections and the current infections over time
PDE = Plot_Data_Epidemic(G_path)
PDE.Accumulative_Infected(ACC_inf)
PDE.Active_Infected()