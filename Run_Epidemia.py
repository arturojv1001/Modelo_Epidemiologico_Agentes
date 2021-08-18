from Nuevas_Clases_otono import Modelo_Epidemia
import matplotlib.pyplot as plt


N          = 10    
steps      = 10
ME  = Modelo_Epidemia(N,steps)


p          = 0.1

G0  = ME.Initial_Graph(1,'ER')


# nodes_size = 30          # [ j for i,j in Go.nodes(data='Nd')]

G_path, ACC_inf     = ME.Epi_Evolution(G0,p)

for t in G_path:
    plt.figure()
    ME.paint_nodes(t)


plt.figure()
plt.plot(ACC_inf)