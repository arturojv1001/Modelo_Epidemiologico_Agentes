from Nuevas_Clases_otono import Modelo_Epidemia
from Nuevas_Clases_otono import Plot_Data_Epidemic
from Nuevas_Clases_otono import Save_Info
import matplotlib.pyplot as plt
import pickle as pkl
import sys
import h5py
import networkx as nx
from tqdm        import tqdm
from subprocess  import call

# Global Parameters
N          = 10    
steps      = 20
# Calling of Classes
ME  = Modelo_Epidemia(N,steps)
# Parameters of the Epidemic
p_i                 = 0.5
p_r                 = 0.5
p_s                 = 0.5
periodo_infeccion   = 3
periodo_inmune      = 2
initial_infected    = 1

Numero_de_Repeticiones = 5


path = "Epidemic_Graphs_N"+str(N)+"/_t"+str(steps)
Save_Info().MakeDir(path)

for run_i in tqdm(range(Numero_de_Repeticiones)):
    call(['python','Run_Epidemic_Simulation.py',str(N),str(steps),str(p_i),str(p_r),str(p_s),str(periodo_infeccion),str(periodo_inmune),str(initial_infected),str(run_i)])
#Save_Info().SaveFiles(path)
Save_Info().SaveFiles(path,make_dir=False,file_format='*.pkl')
    
    
    
    
    
    
    
    
    
    
    
    