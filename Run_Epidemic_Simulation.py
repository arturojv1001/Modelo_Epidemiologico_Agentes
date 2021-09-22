from Nuevas_Clases_otono import Modelo_Epidemia
from Nuevas_Clases_otono import Plot_Data_Epidemic
from Nuevas_Clases_otono import Save_Info
import matplotlib.pyplot as plt
import pickle as pkl
import sys
import h5py
import networkx as nx


# Global Parameters
N          = int(sys.argv[1])    
steps      = int(sys.argv[2])
# Calling of Classes
ME  = Modelo_Epidemia(N,steps)
# Parameters of the Epidemic
p_i                 = float(sys.argv[3])
p_r                 = float(sys.argv[4])
p_s                 = float(sys.argv[5])
periodo_infeccion   = int(sys.argv[6])
periodo_inmune      = int(sys.argv[7])
initial_infected    = int(sys.argv[8])
run                 = int(sys.argv[9])

# we obtain the initial Graph of the epdimedic in the first period of time
G0  = ME.Initial_Graph_atribute(initial_infected,'ER')

# the simulation of the epidemic takes place
G_path, ACC_inf     = ME.Epidemic_Evolution(G0,p_i,periodo_infeccion,p_r,periodo_inmune,p_s)

# the array of graphs is saved into a pkl archive
file_name = "Epidemic_Graphs_N"+str(N)+"_t"+str(steps)+"_run"+str(run)+".pkl"
outfile = open(file_name,'wb')
pkl.dump(G_path,outfile)


path = "Epidemic_Graphs_N"+str(N)+"/_t"+str(steps)
#Save_Info().MakeDir(path)
Save_Info().SaveFiles(path,make_dir=False,file_format='*.pkl')
outfile.close()


