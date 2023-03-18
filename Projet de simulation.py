import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
from scipy import interpolate
from scipy import stats
from scipy.optimize import curve_fit

#quelques fonctions pour tester/vérifier des petites affaires
def has_imaginary_part(arr):
    return np.imag(arr) != 0

def plot_array_bool(arr):
    plt.imshow(arr, cmap='gray', origin='upper')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xticks(range(arr.shape[1]))
    plt.yticks(range(arr.shape[0]))
    plt.show()


# Fonction pour créer les coefficients complexes pour les PML. 
def PML_S(Nx, Ny, layers, w):
    Sx = np.ones((Ny,Nx), dtype=complex)
    Sy = np.ones((Ny,Nx), dtype=complex)
    sigma_x = np.zeros((Ny,Nx), dtype=np.double)
    sigma_y = np.zeros((Ny,Nx), dtype=np.double)

    # initialisation pour les coefficient alpha et alpha*
    sigma_max = 5
    m = 4

    for i in np.arange(0, Ny,1):
        for j in np.arange(0, Nx,1):
            # Pour Sx
            if j>=0 and j<=(layers-1):
                sigma = sigma_max*(((layers)-j)/layers)**m
                sigma_x[i,j] = sigma
                Sx[i,j] = complex(1, (sigma/w))
            elif j<=(Nx-1) and j>=(Nx-layers):
                sigma = sigma_max*((layers-(Nx-j-1))/layers)**m
                sigma_x[i,j] = sigma
                Sx[i,j] = complex(1, (sigma/w))

            if i>=0 and i<=(layers-1):
                sigma = sigma_max*(((layers)-(i))/layers)**m
                sigma_y[i,j] = sigma
                Sy[i,j] = complex(1, (sigma/w))
            elif i<=(Nx-1) and i>=(Nx-layers):
                sigma = sigma_max*((layers-(Nx-i-1))/layers)**m
                sigma_y[i,j] = sigma
                Sy[i,j] = complex(1, (sigma/w))
    
    return(Sx,Sy,sigma_x,sigma_y)
# Fonction pour assembler les blocs de sparses matrices en un gros bloc
def blocks_to_full_matrix(block_matrice_list):
    if len(block_matrice_list) != 16:
        print('Il manque des blocs de matrices')
    else: 
        eq1 = sp.sparse.hstack((block_matrice_list[0],block_matrice_list[1],block_matrice_list[2],block_matrice_list[3]))
        eq2 = sp.sparse.hstack((block_matrice_list[4],block_matrice_list[5],block_matrice_list[6],block_matrice_list[7]))
        eq3 = sp.sparse.hstack((block_matrice_list[8],block_matrice_list[9],block_matrice_list[10],block_matrice_list[11]))
        eq4 = sp.sparse.hstack((block_matrice_list[12],block_matrice_list[13],block_matrice_list[14],block_matrice_list[15]))

        M = sp.sparse.vstack((eq1,eq2,eq3,eq4))
        return(M)



# Initialisation de variable
kappa = 4.2194*10**-10
rho = 1025

# Dimensions de la région de l'océan
Lx=10; #[m]
Ly=10;  #[m]

# Pas initial et facteur de modification
d_initial = 0.1
#fact_ar = np.array([1, 0.5, 0.25, 0.125, 0.0625], dtype=np.double)
fact_ar = np.array([1], dtype=np.double)


# Initialisation d'array pour évaluer la performance
d_ar=np.zeros(fact_ar.size,dtype=np.double) # Array pour le pas de discrétisation
tini_ar=np.zeros(fact_ar.size,dtype=np.double)
tinv_ar=np.zeros(fact_ar.size,dtype=np.double)
mem_ar=np.zeros(fact_ar.size,dtype=np.double)
Tm_ar=np.zeros(fact_ar.size,dtype=np.double)
Err_ar=np.zeros(fact_ar.size-1,dtype=np.double)
d_Err_ar=np.zeros(fact_ar.size-1,dtype=np.double)





ci = -1
for fact in fact_ar:

    # Pas de discrétisation en [m]
    d=d_initial*fact
    d_ar[ci]=d #Array pour le pas de discrétisation

    # Noeuds en x et y
    Nx=int(np.rint(Lx/d+1)); # Nombre de nœuds le long de X
    Ny=int(np.rint(Ly/d+1)); # Nombre de nœuds le long de Y

    # Initialisation de la source ponctuelle
    S=np.zeros((Ny,Nx),dtype=np.double)
    for i in np.arange(1,Ny+1,1):
        y=(i-1)*d
        for j in np.arange(1,Nx+1,1):
            x=(j-1)*d
            # Source mise au milieu du domaine (pour l'instant)
            if i==(Ny/2) and j==(Nx/2):
                S[i-1, j-1] = 1000
    
    # Initialisation de la fréquence
    w = 100000000

    # Matrice des coefficients complexes S
    Sx, Sy, sigma_x, sigma_y = PML_S(Nx, Ny, 10, w)

    # Initialisation d'une liste qui va contenir les matrices de coefficients
    matrice_coefficients_list = []
    # Initialisation des matrices de coefficients pour chacun des inconnus pour la 1ère équation
    M_px_1 = sp.sparse.lil_matrix((Nx*Ny,Nx*Ny),dtype=complex)
    M_py_1 = sp.sparse.lil_matrix((Nx*Ny,Nx*Ny),dtype=complex)
    M_ux_1 = sp.sparse.lil_matrix((Nx*Ny,Nx*Ny),dtype=complex)
    M_uy_1 = sp.sparse.lil_matrix((Nx*Ny,Nx*Ny),dtype=complex) # Matrice sparse "vide"
    b_1=np.zeros((Nx*Ny,1),dtype=np.double)

    M_px_2 = sp.sparse.lil_matrix((Nx*Ny,Nx*Ny),dtype=complex)
    M_py_2 = sp.sparse.lil_matrix((Nx*Ny,Nx*Ny),dtype=complex)
    M_ux_2 = sp.sparse.lil_matrix((Nx*Ny,Nx*Ny),dtype=complex) # Matrice sparse "vide"
    M_uy_2 = sp.sparse.lil_matrix((Nx*Ny,Nx*Ny),dtype=complex)
    b_2=np.zeros((Nx*Ny,1),dtype=np.double)

    M_px_3 = sp.sparse.lil_matrix((Nx*Ny,Nx*Ny),dtype=complex)
    M_py_3 = sp.sparse.lil_matrix((Nx*Ny,Nx*Ny),dtype=complex) # Matrice sparse "vide"
    M_ux_3 = sp.sparse.lil_matrix((Nx*Ny,Nx*Ny),dtype=complex)
    M_uy_3 = sp.sparse.lil_matrix((Nx*Ny,Nx*Ny),dtype=complex) # Matrice sparse "vide"
    b_3=np.zeros((Nx*Ny,1),dtype=np.double)

    M_px_4 = sp.sparse.lil_matrix((Nx*Ny,Nx*Ny),dtype=complex) # Matrice sparse "vide"
    M_py_4 = sp.sparse.lil_matrix((Nx*Ny,Nx*Ny),dtype=complex)
    M_ux_4 = sp.sparse.lil_matrix((Nx*Ny,Nx*Ny),dtype=complex) # Matrice sparse "vide"
    M_uy_4 = sp.sparse.lil_matrix((Nx*Ny,Nx*Ny),dtype=complex)
    b_4=np.zeros((Nx*Ny,1),dtype=np.double)

    px_py_ux_uy=np.zeros((4*Nx*Ny,1),dtype=np.double)


    for i in np.arange(1,Ny+1,1):
        y=(i-1)*d
        for j in np.arange(1,Nx+1,1):
            x=(j-1)*d
            # remplir la ligne pl de la matrice M
            pl=(i-1)*Nx+j

            # Équation 1
            # Noeud exclusivement dans le domaine de simulation (pas les frontières) - px_1
            if ((i>1) and (i<Ny) and (j>1) and (j<Nx)):
                pc=pl
                pc=(i-1)*Nx+j-1;M_px_1[pl-1,pc-1]=-1/(2*d*Sx[i-1, j-1]); # contribution de noeud (i,j-1)
                pc=(i-1)*Nx+j+1;M_px_1[pl-1,pc-1]=1/(2*d*Sx[i-1, j-1]); # contribution de noeud (i,j+1)
                b_1[pl-1]=S[i-1,j-1]
            else: 
                pc=pl
                M_px_1[pl-1,pc-1]=1
                b_1[pl-1]=0
            # Noeud exclusivement dans le domaine de simulation (pas les frontières) - py_1
            if ((i>1) and (i<Ny) and (j>1) and (j<Nx)):
                pc=pl
                pc=(i-1)*Nx+j-1;M_py_1[pl-1,pc-1]=-1/(2*d*Sx[i-1, j-1]); # contribution de noeud (i,j-1)
                pc=(i-1)*Nx+j+1;M_py_1[pl-1,pc-1]=1/(2*d*Sx[i-1, j-1]); # contribution de noeud (i,j+1)
                b_1[pl-1]=S[i-1,j-1]
            else: 
                pc=pl
                M_py_1[pl-1,pc-1]=1
                b_1[pl-1]=0
            # Noeud exclusivement dans le domaine de simulation (pas les frontières) - ux_1
            if ((i>1) and (i<Ny) and (j>1) and (j<Nx)):
                pc=pl
                M_ux_1[pl-1,pc-1]=complex(sigma_x[i-1,j-1]*rho,w*rho); # contribution de noeud (i,j)
                b_1[pl-1]=S[i-1,j-1]
            else: 
                pc=pl
                M_ux_1[pl-1,pc-1]=1
                b_1[pl-1]=0

            # Équation 2
            # Noeud exclusivement dans le domaine de simulation (pas les frontières) - px_2
            if ((i>1) and (i<Ny) and (j>1) and (j<Nx)):
                pc=pl
                pc=(i-2)*Nx+j;M_px_2[pl-1,pc-1]=-1/(2*d*Sy[i-1, j-1]); # contribution de noeud (i-1,j)
                pc=(i)*Nx+j;M_px_2[pl-1,pc-1]=1/(2*d*Sy[i-1, j-1]); # contribution de noeud (i+1,j)
                b_2[pl-1]=S[i-1,j-1]
            else: 
                pc=pl
                M_px_2[pl-1,pc-1]=1
                b_2[pl-1]=0
            # Noeud exclusivement dans le domaine de simulation (pas les frontières) - py_2
            if ((i>1) and (i<Ny) and (j>1) and (j<Nx)):
                pc=pl
                pc=(i-2)*Nx+j;M_py_2[pl-1,pc-1]=-1/(2*d*Sy[i-1, j-1]); # contribution de noeud (i-1,j)
                pc=(i)*Nx+j;M_py_2[pl-1,pc-1]=1/(2*d*Sy[i-1, j-1]); # contribution de noeud (i+1,j)
                b_2[pl-1]=S[i-1,j-1]
            else: 
                pc=pl
                M_py_2[pl-1,pc-1]=1
                b_2[pl-1]=0
            # Noeud exclusivement dans le domaine de simulation (pas les frontières) - uy_2
            if ((i>1) and (i<Ny) and (j>1) and (j<Nx)):
                pc=pl
                M_uy_2[pl-1,pc-1]=complex(sigma_y[i-1,j-1]*rho,w*rho); # contribution de noeud (i,j)
                b_2[pl-1]=S[i-1,j-1]
            else: 
                pc=pl
                M_uy_2[pl-1,pc-1]=1
                b_2[pl-1]=0

            # Équation 3
            # Noeud exclusivement dans le domaine de simulation (pas les frontières) - ux_3
            if ((i>1) and (i<Ny) and (j>1) and (j<Nx)):
                pc=pl
                pc=(i-1)*Nx+j-1;M_ux_3[pl-1,pc-1]=-1/(2*d*Sx[i-1, j-1]); # contribution de noeud (i,j-1)
                pc=(i-1)*Nx+j+1;M_ux_3[pl-1,pc-1]=1/(2*d*Sx[i-1, j-1]); # contribution de noeud (i,j+1)
                b_3[pl-1]=0 #PT 0? 
            else: 
                pc=pl
                M_ux_3[pl-1,pc-1]=1
                b_3[pl-1]=0
            # Noeud exclusivement dans le domaine de simulation (pas les frontières) - px_3
            if ((i>1) and (i<Ny) and (j>1) and (j<Nx)):
                pc=pl
                M_px_3[pl-1,pc-1]=complex(sigma_x[i-1,j-1]*kappa,w*kappa); # contribution de noeud (i,j)
                b_3[pl-1]=0
            else: 
                pc=pl
                M_px_3[pl-1,pc-1]=1
                b_3[pl-1]=0


            # Équation 4
            # Noeud exclusivement dans le domaine de simulation (pas les frontières) - uy_4
            if ((i>1) and (i<Ny) and (j>1) and (j<Nx)):
                pc=pl
                pc=(i-2)*Nx+j;M_uy_4[pl-1,pc-1]=-1/(2*d*Sy[i-1, j-1]); # contribution de noeud (i-1,j)
                pc=(i)*Nx+j;M_uy_4[pl-1,pc-1]=1/(2*d*Sy[i-1, j-1]); # contribution de noeud (i+1,j)
                b_4[pl-1]=0
            else: 
                pc=pl
                M_uy_4[pl-1,pc-1]=1
                b_4[pl-1]=0
            # Noeud exclusivement dans le domaine de simulation (pas les frontières) - py_4
            if ((i>1) and (i<Ny) and (j>1) and (j<Nx)):
                pc=pl
                M_py_4[pl-1,pc-1]=complex(sigma_y[i-1,j-1]*kappa,w*kappa); # contribution de noeud (i,j)
                b_4[pl-1]=0
            else: 
                pc=pl
                M_py_4[pl-1,pc-1]=1
                b_4[pl-1]=0


    matrice_coefficients_list.extend((M_px_1, M_py_1, M_ux_1, M_uy_1, 
                                      M_px_2, M_py_2, M_ux_2, M_uy_2, 
                                      M_px_3, M_py_3, M_ux_3, M_uy_3,
                                      M_px_4, M_py_4, M_ux_4, M_uy_4))
    M=blocks_to_full_matrix(matrice_coefficients_list)

    b = np.vstack((b_1, b_2, b_3, b_4))

    M=sp.sparse.lil_matrix.tocsc(M) 
    px_py_ux_uy=sp.sparse.linalg.spsolve(M,b)

    P=px_py_ux_uy[0:Nx*Ny] + px_py_ux_uy[Nx*Ny:2*Nx*Ny]
    Tr=np.reshape(P,(Ny,Nx),order='C')

plt.figure(1)
plt.pcolor(np.arange(0,Nx,1)*d,np.arange(0,Ny,1)*d,Tr.real);
plt.colorbar(mappable=None, cax=None, ax=None);
plt.xlabel('x [m]')    
plt.ylabel('y [m]')
plt.show()


    

    

    







