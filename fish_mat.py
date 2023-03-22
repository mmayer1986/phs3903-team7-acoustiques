##### Packages #######
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
from scipy import interpolate
from scipy import stats
from scipy.optimize import curve_fit
from scipy import ndimage

#### Function to make a matrix

### Cette fonction fait une matrice de zéros
def matrix_maker(x=100, y =100):
    return np.zeros([x,y])

### Cette fonction place un point à un endroit désirer
def dot_placer(matrice, x=50, y=50):
    matrice[x,y] = 1
    return matrice


###Cette fonction montre visuellement la position de 1 point
def show_dot(x_len=100,y_len=100, x_pos=50, y_pos=50):
    mat = matrix_maker(x_len,y_len)
    mat = dot_placer(mat, x_pos,y_pos)
    plt.imshow(mat)
    return mat


## Cette fonction crée des cercles pleins
def circular_fish_maker(matrice, x_origin, y_origin, radius):
    for y in range(len(matrice)):
        for x in range(len(matrice[0])):
            #Ici, l'équation d'un cercle est utilisé. Si x et y sont dans le rayon, la valeur de 1 est attribué
            if (y - y_origin)**2 <= radius**2 - (x - x_origin)**2:
                if (x - x_origin)**2 <= radius**2 - (y - y_origin)**2:
                    matrice[x,y] = 1
                ## Et zéro sinon
                else:
                    matrice[x,y] = 0
            else:
                matrice[x,y] = 0
    return matrice


## Cette fonction montre les cercles
def show_circle_fish(x_len=100,y_len=100, x_pos=50, y_pos=50, radius = 5):
    mat = matrix_maker(x_len,y_len)
    mat = circular_fish_maker(mat, x_pos, y_pos, radius)
    plt.imshow(mat)
    return mat


## Fonction pareil au précédent mais avec des ovales
def oval_fish_maker(matrice, x_origin, y_origin, major_axis_size):
    minor_axis_size = major_axis_size/2
    for y in range(len(matrice)):
        for x in range(len(matrice[0])):
            if (y - y_origin)**2/major_axis_size**2 <= 1 - (x - x_origin)**2/minor_axis_size**2:
                if (x - x_origin)**2/minor_axis_size**2 <= 1 - (y - y_origin)**2/major_axis_size**2:
                    matrice[x,y] = 1
                
                else:
                    matrice[x,y] = 0
            else:
                matrice[x,y] = 0
    return matrice

### Pour voir ton poisson oval...
def show_oval_fish(x_len=500,y_len=500, x_pos=250, y_pos=250, major_axis_size = 50):
    mat = matrix_maker(x_len,y_len)
    mat = oval_fish_maker(mat, x_pos, y_pos, major_axis_size)
    plt.imshow(mat)
    return mat

## Ce filtre gaussien permet de rendre les edges un peu plus flou et les ovales plus ovales. L'écart type ne peut pas être très grand
def filterd_fish(matrice, standard_deviation=0.5):
    mat = ndimage.gaussian_filter(matrice, standard_deviation)
    # plt.imshow(mat)
    return mat

## Cette fonction donne des positions aléatoires pour un nombre de positions désirés
def random_position(matrice, number_of_positions = 1, x_dist_from_PML=5, y_dist_from_PML=5):
    x_len = len(matrice[0]) - x_dist_from_PML #le dist from PML est la distance des limites de l'espace pour éviter les PMLs et zones de comportement étrange
    y_len = len(matrice) - y_dist_from_PML
    
    #fonction random number generator (rng)
    rng = np.random.default_rng()
    
    #On génére les numéros aléatoires
    x_rand = rng.choice(x_len, number_of_positions, replace=False)
    y_rand = rng.choice(y_len, number_of_positions, replace=False)
    return x_rand, y_rand


## Cette fonction place un oval à un nombre de positions aléatoires désiré. Taille de poisson fixe
def rand_fish_placer(x_len = 500, y_len = 500, number_of_fish = 1, x_from_PML = 5, y_from_PML=5, taille_poisson = 5):
    
    #matrix_maker fais une matrice de zéro
    mat_final = matrix_maker(x_len,y_len)

    #les positions aléatoires sont trouvées
    x_fish_position, y_fish_position = random_position(mat_final, number_of_positions=number_of_fish, x_dist_from_PML=x_from_PML, y_dist_from_PML=y_from_PML)
    
    #On itère pour le nombre de poisson
    for i in range(number_of_fish):
        mat_blank = matrix_maker(x_len,y_len)
        #un poisson oval est créé à chaque itération
        oval_fish_maker(mat_blank, x_fish_position[i], y_fish_position[i], major_axis_size = taille_poisson)
        #et ajouté à la matrice finale
        mat_final += mat_blank
    return mat_final

## Comme le précédent mais choisie aussi des tailles aléatoires
def rand_fish_size_and_pos(x_len = 500, y_len = 500, number_of_fish = 1, x_from_PML = 5, y_from_PML=5, taille_poisson = [3,5]):
    
    mat_final = matrix_maker(x_len,y_len)
    x_fish_position, y_fish_position = random_position(mat_final, number_of_positions=number_of_fish, x_dist_from_PML=x_from_PML, y_dist_from_PML=y_from_PML)
    
    #fonction random number generator (rng)
    rng = np.random.default_rng()
    #Des tailles aléatoires généré du range donné le défaut étant de 3 à 5 (la taille est la longeur de l'axe majeur de l'ellipse)
    tailles_aléatoires = rng.choice(np.arange(taille_poisson[0], taille_poisson[1]+1, 1), number_of_fish, replace=True)

    for i in range(number_of_fish):
        mat_blank = matrix_maker(x_len,y_len)
        oval_fish_maker(mat_blank, x_fish_position[i], y_fish_position[i], major_axis_size = tailles_aléatoires[i])
        mat_final += mat_blank
    return mat_final



##Ce code fais des rotations d'une angle désiré se basant sur la taille d'un poisson
def poisson_rotater(rotation=30, taille_poisson = 6, filtre=False):
    mat = matrix_maker(taille_poisson*2+2, taille_poisson*2+2)
    centre = int(len(mat)/2)
    mat = oval_fish_maker(mat, centre, centre, taille_poisson)
    mat_rot = ndimage.interpolation.rotate(mat, rotation,reshape=False)
    
    for row in range(len(mat_rot)):
        for col in range(len(mat_rot[0])):
            if mat_rot[row][col] <= 0.06:
                mat_rot[row][col] = 0
            else:
                mat_rot[row][col] = 1

    if filtre == True:
        filt_mat_rot = filterd_fish(mat_rot, standard_deviation=0.5)
        return filt_mat_rot
    else:
        return mat_rot


def rotated_fish_placer(matrice, x, y, rotation=30, taille_poisson = 6, filtre=False):
    A = matrice
    B = poisson_rotater(rotation, taille_poisson, filtre)
    r,c = x,y # coordonéées
    A[r:r+B.shape[0], c:c+B.shape[1]] += B
    # plt.imshow(A)
    return A


#Cette fonction génére une matrice avec des tailles, positions et orientations de poisson de façon aléatoire
#C'est important que la distance des PMLs soit supérieur à deux fois la taille maximal des poissons
#The only thing that needs to be fixed is overlapping fish
def rand_fish_size_pos_rot(x_len = 500, y_len = 500, number_of_fish = 1, x_from_PML = 10, y_from_PML=10, taille_poisson = [3,5], rotations = [0,360], pas_angle = 30, filtre = True):
    
    mat_final = matrix_maker(x_len,y_len)
    x_fish_position, y_fish_position = random_position(mat_final, number_of_positions=number_of_fish, x_dist_from_PML=x_from_PML, y_dist_from_PML=y_from_PML)
    
    #fonction random number generator (rng)
    rng = np.random.default_rng()
    #Des tailles aléatoires généré du range donné le défaut étant de 3 à 5 (la taille est la longeur de l'axe majeur de l'ellipse)
    tailles_aléatoires = rng.choice(np.arange(taille_poisson[0], taille_poisson[1]+1, 1), number_of_fish, replace=True)
    angles_aléatoires = rng.choice(np.arange(rotations[0], rotations[1]+1, pas_angle), number_of_fish, replace=True)

    for i in range(number_of_fish):
        mat_blank = matrix_maker(x_len,y_len)
        rotated_fish_placer(mat_blank, x_fish_position[i], y_fish_position[i], rotation=angles_aléatoires[i], taille_poisson = tailles_aléatoires[i], filtre=filtre)
        mat_final += mat_blank
    return mat_final