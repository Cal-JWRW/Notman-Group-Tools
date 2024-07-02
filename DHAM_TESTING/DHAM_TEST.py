import numpy as np
from DHAM import Build_Matrices
from DHAM import Optimize_Dhamed
from DHAM import DHAM
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
Temp = 303.15
with open('px.dat') as f:
    px = [line.rstrip() for line in f]

states = np.linspace(0, 40.0, 400)
Biasing_Pot = 5000*0.00239
trj_matrices=[]
bias_pot_matrices = []
for i in tqdm(px):
    trj = np.abs(np.asarray(pd.read_table(i, skiprows=17, delim_whitespace=True, header=None)[1]))
    trj_matrices.append(Build_Matrices.Count_Matrix(trj*10, 45, states))
    bias_pot_matrices.append(Build_Matrices.Bias_Matrix(trj*10, Biasing_Pot, states, Temp))
bias_pot_matrices = np.asarray( bias_pot_matrices)
MM = DHAM.Construct_MM(trj_matrices, bias_pot_matrices.T, states, Temp)
print(MM)

